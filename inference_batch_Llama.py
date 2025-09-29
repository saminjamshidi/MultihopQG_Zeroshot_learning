import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from types import SimpleNamespace
from GAT import AttentionLayer
from json import dumps

# ========== Config ==========
model_id = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/home/mahdiani/mistral/"
SAVE_PATH = "./generated_questions_batched_llama.jsonl"
BATCH_SIZE = 8

# ========== Load Models ==========
print("Loading LLaMA model and tokenizer...")
llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=True)
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

print("Loading BART encoder for embeddings...")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", use_safetensors=True).to(DEVICE)

for param in bart_model.model.encoder.parameters():
    param.requires_grad = False

# ========== Helper Functions ==========
def get_embedding(entity_texts):
    inputs = bart_tokenizer(entity_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = bart_model.model.encoder(input_ids)
    return outputs.last_hidden_state.mean(dim=1)

def build_zero_shot_prompt(document, answer, entities=None):
    if entities:
        entity_str = "; ".join(entities)
        user_msg = (
            "[CONTEXT]   " + document + "\n"
            "[ANSWER]    " + answer   + "\n"
            "[IMPORTANT ENTITIES] " + entity_str + "\n"
            "[QUESTION]"
        )
    else:
        user_msg = (
            "[CONTEXT]   " + document + "\n"
            "[ANSWER]    " + answer   + "\n"
            "[QUESTION]"
        )

    messages = [
        {
            "role": "system",
            "content": "You are an expert question-writer. Generate a complex multi-hop "
                       "question whose answer is the provided answer."
        },
        {"role": "user", "content": user_msg},
    ]

    # Converts the list-of-dicts into:
    # <s>[INST] <<SYS>>…<</SYS>> … [/INST]
    return llama_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # inserts the assistant tag
        tokenize=False
    )

# ========== GAT Model ==========
class BinaryClassifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_head, config):
        super().__init__()
        self.gat_leyer = AttentionLayer(in_dim, out_dim, n_head, config)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(80 * 1024, 80)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, node_features, adjacency_matrices, masks):
        x = self.gat_leyer(node_features, adjacency_matrices, masks)
        return self.sigmoid(self.fc(self.flatten(x)))

print("Loading GAT model...")
config = SimpleNamespace(gnn_drop=0.5, q_attn=False, hidden_dim=8, n_type=4, q_update=False, input_dim=20)
gat_model = BinaryClassifier(in_dim=1024, out_dim=1024, n_head=4, config=config).to(DEVICE)
gat_model.load_state_dict(torch.load(DATA_DIR + "fine_tuned_gat.pth", map_location=DEVICE))
gat_model.eval()

# ========== Load Data ==========
print("Loading data...")
with open(DATA_DIR + "concate_data_2.pkl", "rb") as f:
    all_data = pickle.load(f)
with open(DATA_DIR + "test_data_v10.pkl", "rb") as f:
    test_indices = pickle.load(f)
test_data = [all_data[i] for i in test_indices]

# ========== Main Batched Inference ==========
print("Generating questions in batches...")
with open(SAVE_PATH, "w") as out_file:
    for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
        batch = test_data[i:i + BATCH_SIZE]

        documents = [sample["document"] for sample in batch]
        answers = [sample["answer"] for sample in batch]
        questions_gt = [sample.get("Question", "") for sample in batch]
        entity_lists = [[e[2] for e in sample["entity_spans"]] for sample in batch]
        entity_flat = [e for group in entity_lists for e in group]

        # Get entity embeddings
        entity_embeddings = get_embedding(entity_flat)

        # Split and pad
        batched_entity_tensor = []
        idx = 0
        for entity_set in entity_lists:
            cur = entity_embeddings[idx:idx + len(entity_set)]
            cur = F.pad(cur, (0, 0, 0, 80 - len(cur))) if len(cur) < 80 else cur[:80]
            batched_entity_tensor.append(cur)
            idx += len(entity_set)

        entity_tensor_batch = torch.stack(batched_entity_tensor).to(DEVICE)
        adjacency_batch = torch.stack([
            sample["Adg"] if isinstance(sample["Adg"], torch.Tensor) else torch.tensor(sample["Adg"])
            for sample in batch
        ]).to(DEVICE)

        mask_batch = torch.stack([
            sample["mask"] if isinstance(sample["mask"], torch.Tensor) else torch.tensor(sample["mask"])
            for sample in batch
        ]).to(DEVICE)

        with torch.no_grad():
            output = gat_model(entity_tensor_batch, adjacency_batch, mask_batch)
            predicted_mask = (output > 0.5).int()

        prompts_with_entities = []
        prompts_without_entities = []
        kept_entities = []

        for b, entity_set in enumerate(entity_lists):
            important = [entity_set[i] for i in range(min(len(entity_set), 80)) if predicted_mask[b, i] == 1]
            kept_entities.append(important)
            prompts_with_entities.append(build_zero_shot_prompt(documents[b], answers[b], important))
            prompts_without_entities.append(build_zero_shot_prompt(documents[b], answers[b], entities=None))

        inputs_with = llama_tokenizer(prompts_with_entities, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        inputs_without = llama_tokenizer(prompts_without_entities, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        gen_cfg = dict(
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=llama_tokenizer.eos_token_id   # avoids pad-token warnings
        )

        with torch.no_grad():
            outs_with    = llama_model.generate(**inputs_with,  **gen_cfg)
            outs_without = llama_model.generate(**inputs_without, **gen_cfg)

        decoded_with    = llama_tokenizer.batch_decode(outs_with,    skip_special_tokens=True)
        decoded_without = llama_tokenizer.batch_decode(outs_without, skip_special_tokens=True)

        for j in range(len(decoded_with)):
            out_file.write(dumps({
                "context": documents[j],
                "answer": answers[j],
                "entities": kept_entities[j],
                "generated_with_entities": decoded_with[j].split("[QUESTION]")[-1].strip(),
                "generated_without_entities": decoded_without[j].split("[QUESTION]")[-1].strip(),
                "ground_truth_question": questions_gt[j]
            }) + "\n")

print(f"Done. Results saved to {SAVE_PATH}")

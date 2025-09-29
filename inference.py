import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from types import SimpleNamespace
from GAT import AttentionLayer
from json import dumps

# ========== Model Config ==========
model_id = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/home/mahdiani/mistral/"
SAVE_PATH = "./generated_questions.jsonl"

# ========== Load Models ==========
print("Loading LLaMA model and tokenizer...")
llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=True)
llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

print("Loading BART encoder for embeddings...")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", use_safetensors=True).to(DEVICE)

def freeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = False
freeze_encoder(bart_model)

# ========= Embedding Extraction =========
def get_embedding(entity_texts):
    inputs = bart_tokenizer(entity_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = bart_model.model.encoder(input_ids)
    return outputs.last_hidden_state.mean(dim=1)

# ========= GAT Model =========
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

# ========= Load Data =========
print("Loading data...")
with open(DATA_DIR + "concate_data_2.pkl", "rb") as f:
    all_data = pickle.load(f)
with open(DATA_DIR + "test_data_v10.pkl", "rb") as f:
    test_indices = pickle.load(f)
test_data = [all_data[i] for i in test_indices]

# ========= Prompting and Inference =========
def build_zero_shot_prompt(document, answer, entities):
    return f"""You are an expert question-writer. Given a passage, an answer, and a list of important entities, generate a complex multi-hop question whose answer is the provided answer.

[ENTITIES]  {' ; '.join(entities)}
[CONTEXT]   {document}
[ANSWER]    {answer}
[QUESTION]"""

def generate_question(prompt):
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    output = llama_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return llama_tokenizer.decode(output[0], skip_special_tokens=True).split("[QUESTION]")[-1].strip()

# ========= Main Loop =========
print("Generating questions...")
with open(SAVE_PATH, "w") as out_file:
    for sample in tqdm(test_data):
        document = sample["document"]
        answer = sample["answer"]
        entity_list = [e[2] for e in sample["entity_spans"]]
        
        adjacency_matrix = sample["Adg"].unsqueeze(0).to(DEVICE) if isinstance(sample["Adg"], torch.Tensor) \
            else torch.tensor(sample["Adg"]).unsqueeze(0).to(DEVICE)
        mask = sample["mask"].unsqueeze(0).to(DEVICE) if isinstance(sample["mask"], torch.Tensor) \
            else torch.tensor(sample["mask"]).unsqueeze(0).to(DEVICE)

        entity_tensor = get_embedding(entity_list)
        if entity_tensor.size(0) < 80:
            entity_tensor = F.pad(entity_tensor, (0, 0, 0, 80 - entity_tensor.size(0)))
        entity_tensor = entity_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = gat_model(entity_tensor, adjacency_matrix, mask)
            predicted_mask = (output > 0.5).int()[0]

        important_entities = [
            entity_list[i] for i in range(min(len(entity_list), predicted_mask.size(0)))
            if predicted_mask[i] == 1
        ]

        # print(f"\n[INFO] Sample ID {sample.get('Id', 'N/A')} — Document Length: {len(sample['document'])}")
        # print(f"[INFO] Entity List: {entity_list}")

        # print(f"[INFO] GAT prediction: {predicted_mask.tolist()}")
        # print(f"[INFO] Important entities: {important_entities}")

        # if not important_entities:
        #     print(f"[WARN] No important entities found for sample ID {sample.get('Id', 'N/A')}")
        #     continue


        prompt = build_zero_shot_prompt(document, answer, important_entities)
        # print(f"[PROMPT]\n{prompt[:300]}...\n")

        question = generate_question(prompt)
        # print(f"[QUESTION GENERATED]\n{question}\n")
        # if not question.strip():
        #     print("[SKIP] Empty question — skipping.")
        #     continue

        out_file.write(dumps({
            "context": document,
            "answer": answer,
            "entities": important_entities,
            "generated_question": question
        }) + "\n")

print(f"Done. Questions saved to {SAVE_PATH}")

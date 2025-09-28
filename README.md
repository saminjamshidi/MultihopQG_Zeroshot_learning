# Graph-Guided Prompting for Zero-Shot Multi-Hop Question Generation
**Gains without Fine-Tuning, Limits without Adaptation**

> Samin Jamshidi · Morteza Mahdiani  
> ICML 2025 (PMLR 267) — Vancouver, Canada

---

## 🔍 Overview

This work introduces a **modular, zero-shot** framework for **multi-hop question generation (MHQG)**. A lightweight **Graph Attention Network (GAT)** learns which entities and relations in a passage–answer pair are most indicative of the reasoning chain. The selected entities are woven back into the context to form an **entity-enriched prompt**, which is fed to **frozen, inference-only LLMs** (e.g., **Llama-2-Chat-7B**, **DeepSeek-Coder-Instruct-6.7B**). The decoupled design improves MHQG without any fine-tuning of the generator, achieving consistent gains on **HotpotQA** while keeping compute costs low. :contentReference[oaicite:0]{index=0}

---

## ✨ Key Contributions

- **Zero-shot MHQG via graph-guided prompting**: A trained GAT highlights salient entities and their relations and injects them into the prompt; the LLM remains **parameter-locked**. :contentReference[oaicite:1]{index=1}  
- **Plug-and-play reasoning module**: Separates **structured reasoning** from language modeling—easy to pair with newer or larger LLMs without architectural changes or joint training. :contentReference[oaicite:2]{index=2}  
- **Empirical gains** on HotpotQA: Improves BLEU, ROUGE-L, and METEOR over plain zero-shot prompting for both Llama and DeepSeek backbones. :contentReference[oaicite:3]{index=3}  
- **Clear limits**: Still trails fully fine-tuned task-specific systems, suggesting graph guidance is **complementary** to end-to-end adaptation. :contentReference[oaicite:4]{index=4}

---

## 🧠 Method at a Glance

1. **Entity Graph**: Build a graph from named entities with edges for sentence-level co-occurrence, paragraph/title association, and cross-paragraph coreference; initialize nodes with contextual embeddings. :contentReference[oaicite:5]{index=5}  
2. **GAT Reasoning**: A multi-head GAT (hidden size 128) scores entities by their importance for multi-hop reasoning, trained with a binary objective on entities appearing in both supporting facts and reference question span. :contentReference[oaicite:6]{index=6}  
3. **Entity-Enriched Prompt**: Concatenate `[Context; Answer; Selected-Entities]` as the **enriched prompt**. :contentReference[oaicite:7]{index=7}  
4. **Zero-Shot Generation**: Feed the prompt to a **frozen LLM** (Llama-2-Chat-7B or DeepSeek-Coder-Instruct-6.7B) with greedy decoding—**no fine-tuning**. :contentReference[oaicite:8]{index=8}

> The separation keeps the reasoning module small and adaptable, while reusing the LLM’s pretrained fluency. :contentReference[oaicite:9]{index=9}

---

## 📊 Results (HotpotQA)

**Automatic metrics** (higher is better):

| Model                | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR |
|----------------------|:------:|:------:|:------:|:------:|:-------:|:------:|
| **GAT + Llama**      | 10.77  | 4.66   | 2.55   | 1.68   | 14.17   | 16.53  |
| Llama                | 10.05  | 4.19   | 2.38   | 1.61   | 12.62   | 15.02  |
| **GAT + DeepSeek**   | 16.98  | 8.36   | 4.75   | 3.11   | 21.87   | 26.59  |
| DeepSeek             | 16.24  | 7.95   | 4.61   | 3.05   | 19.94   | 25.06  |

The **graph-guided prompt** consistently outperforms plain zero-shot prompting across all metrics for both backbones. :contentReference[oaicite:10]{index=10}

---

## 🧪 Reproducing (High-Level)

Although the paper evaluates with open LLMs, the framework is **model-agnostic**:

1. **Prepare data** (e.g., HotpotQA) with passage, answer, supporting facts, and reference question. :contentReference[oaicite:11]{index=11}  
2. **Train GAT** for entity selection using positive labels from (supporting-facts ∩ question-span). Monitor validation F1 (peaks ~epoch 8 in our runs). :contentReference[oaicite:12]{index=12}  
3. **Build prompts**: concatenate `[C; A; E_sub]` using entities above a selection threshold. :contentReference[oaicite:13]{index=13}  
4. **Generate** with a frozen LLM (e.g., Llama-2-Chat-7B / DeepSeek-Coder-Instruct-6.7B). Evaluate with BLEU/ROUGE-L/METEOR. :contentReference[oaicite:14]{index=14}

> The same reasoning module can be paired with newer LLMs as longer context windows become available. :contentReference[oaicite:15]{index=15}

---

## ⚠️ Limitations & Future Work

- Performance remains below **fully fine-tuned** generators; graph guidance is **complementary** to adaptation.  
- Results depend on **entity extraction** quality and a simple linear prompt format `[C; A; E_sub]`.  
- Future directions: integrate **few-shot exemplars**, optimize prompt templates for reasoning, and scale to **larger/backbone-diverse** LLMs. :contentReference[oaicite:16]{index=16}

---

## 📣 Citation

If you use this work, please cite:

```bibtex
@inproceedings{jamshidi2025graphguided,
  title     = {Graph-Guided Prompting for Zero-Shot Multi-Hop Question Generation: Gains without Fine-Tuning, Limits without Adaptation},
  author    = {Samin Jamshidi and Morteza Mahdiani},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {267},
  year      = {2025},
  address   = {Vancouver, Canada}
}

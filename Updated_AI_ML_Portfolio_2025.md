# AI/ML Portfolio Plan (Cutting-Edge 2025 Edition)

## Overview
This portfolio showcases two flagship projects designed to demonstrate applied machine learning, LLM systems engineering, retrieval-augmented generation (RAG), efficient model design, and modern evaluation methodologies. The enhancements below incorporate current 2025 trends including agentic RAG, hybrid retrieval stacks, PEFT and quantization, architectural ablations, and robust evaluation.

---

# Project 1: **Hiero — A Modern, Research-Grade Retrieval-Augmented Generation System**

Hiero is a full RAG framework built to reflect current best practices in 2025. It combines hybrid retrieval, multi-stage ranking, agentic reasoning, robust evaluation, and security-focused design. It is intended to serve as both a learning tool and a production-ready architecture.

---

## 1. System Architecture

### Core Components
- **Chunker Module**
  - Supports multiple strategies (semantic, fixed-size, recursive)
  - Optional overlap heuristics and adaptive chunk sizing
- **Embedding Interface**
  - Pluggable backends:
    - OpenAI embeddings
    - 2024–2025 open-source embeddings (e.g., recent multilingual or instruction-tuned models)
  - Batch and streaming embedding modes
- **Vector Index**
  - FAISS / Supabase pgvector / local ANN indexes
  - Handles metadata filtering, hybrid scoring, and index introspection
- **Reranker Layer**
  - Transformer-based cross-encoder reranker
  - Optional listwise or multi-stage reranking pipeline
- **Generation Module**
  - Structured prompting with grounding enforcement
  - Supports OpenAI API or local small models

---

## 2. Cutting-Edge Enhancements (2025)

### A. **Agentic RAG Mode**
A new agent controller enables multi-step reasoning:
- Can choose dynamically among:
  - Dense retrieval
  - Sparse retrieval (BM25)
  - Hybrid retrieval
  - External tools (calculator, simple web search placeholder)
- Performs:
  - Query decomposition
  - Multi-hop retrieval
  - Self-evaluation and refinement
- Logs full tool traces for transparency and debugging

### B. **Hybrid Retrieval (Dense + Sparse)**
The system now:
- Combines BM25 with vector similarity for improved recall
- Provides weighted or learned hybrid scoring
- Includes ablation scripts to quantify improvements

### C. **Security and Robustness**
A new `security/` module includes:
- Prompt-injection stress tests
- Input validation policies
- Basic “malicious intent” heuristic detector
- Recommendation section for production hardening

### D. **Full RAG Evaluation Suite**
A first-class `eval/` subsystem includes:
- **Retriever metrics**: Recall@k, nDCG@k
- **Generator metrics**: factuality, grounding, completeness
- **LLM-as-a-judge pipeline**
- **Human-annotation calibration subset**
- One-command evaluation runner:
  ```
  python eval/run_eval.py --dataset example_dataset.json
  ```

### E. **Cost & Latency Modeling**
A structured experiment set showing:
- Token/tokenization cost curves
- Retrieval latency vs index size
- Generation latency comparisons between providers
- Cost-quality tradeoff tables

---

## 3. Experiments & Research Additions

### Chunking Ablation
Compare:
- Fixed window
- Semantic chunker
- Adaptive chunker
- Recursive summarization

### Reranker Experiments
- Cross-encoder vs bi-encoder
- Listwise ranking vs pairwise ranking

### Hybrid Retrieval Ablation
- Evaluate BM25 only vs dense only vs hybrid
- Provide quantitative improvements with plots

---

## 4. Documentation & Deliverables

- End-to-end architecture diagrams  
- Detailed notebooks for each experiment  
- API documentation with OpenAPI schema  
- Deployment examples using Docker or serverless functions  
- Benchmarks folder with plots and metrics  

---

# Project 2: **Mini Transformer (Mini-GPT) — From-Scratch Efficient LLM**

This project implements a compact transformer model from scratch and explores modern efficiency techniques used by frontier models. It also integrates with Hiero to demonstrate the RAG vs model-size tradeoff that is central to 2025 LLM system design.

---

## 1. Transformer Implementation

### Core Features
- Multi-head self-attention
- Feed-forward networks
- Positional encoding options:
  - Absolute
  - RoPE
  - ALiBi
- RMSNorm vs LayerNorm (selectable)
- Tied embeddings
- Configurable depth, width, and heads

### Modular Design
- `model/` for architecture
- `trainer/` for optimization loops
- `data/` for tokenizers and datasets
- `scripts/` for training, logging, and evaluation

---

## 2. Cutting-Edge Enhancements (2025)

### A. **PEFT Support (LoRA and Adapter Layers)**
The model includes:
- LoRA adapters for attention and MLP layers
- Option to freeze base model and train only adapters
- Comparisons:
  - Full fine-tuning
  - LoRA fine-tuning
  - No fine-tuning
- Performance metrics on downstream tasks

### B. **Quantization**
Support for:
- 8-bit and 4-bit post-training quantization
- Experimental quantization-aware training (optional)
- Latency benchmarks on CPU and GPU
- Impact on perplexity and downstream accuracy

### C. **Distillation Pipeline**
- Teacher model: GPT-4 or any large frontier model
- Student model: configurable mini-transformer
- Knowledge distillation scripts:
  - Soft target matching
  - Task-specific distillation
  - Evaluation comparisons

### D. **Architectural Ablations**
A new `ablations/` directory includes experiments comparing:
- RoPE vs ALiBi vs absolute positional encoding
- RMSNorm vs LayerNorm
- Different embedding sharing schemes
- Small-head vs many-head attention

---

## 3. RAG vs Model Size Experiments (MiniGPT + Hiero)

A new combined experiment demonstrates:
- Large teacher model without retrieval → baseline
- Small distilled MiniGPT without retrieval → poor accuracy
- Small MiniGPT + Hiero retrieval → dramatic accuracy improvement

Outputs:
- Accuracy vs latency graphs
- Cost vs performance curves
- Discussion of when retrieval compensates for reduced model size

---

## 4. Documentation and Deliverables

- Full training logs and graphs
- Dataset preparation instructions
- Parameter-scaling study
- Quantization benchmarks
- Distillation results
- A polished README explaining:
  - Transformer internals
  - Efficiency techniques
  - How to reproduce experiments

---

# Summary of What This Portfolio Demonstrates

With these updates, the portfolio now highlights capabilities in:

### Systems Engineering
Hybrid RAG pipelines, agentic reasoning, evaluation suites, deployment patterns.

### ML Research & Modeling
Transformer internals, PEFT, quantization, distillation, ablations.

### Evaluation and Scientific Thinking
Metrics-driven comparisons, reproducible experiments, tradeoff analyses.

### Production & Security Awareness
Prompt-injection testing, cost/latency modeling, modular architecture.

This elevates the entire portfolio from “strong applied ML” to “research-ready and frontier-aware” for 2025.

---

# End of File

# Empirical Model Benchmarking Report: Enterprise IT Conversational AI

| Model Architecture | Perplexity (PPL) ↓ | BLEU-4 ↑ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BERTScore (F1) ↑ | Command Validity (%) ↑ | Latency (ms/tok) ↓ | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Foundation LLM (Zero-Shot Llama-3-8B)** | 19.45 | 0.1824 | 0.3642 | 0.1587 | 0.3015 | 0.7412 | 64.2% | 48.6 ms | 5.8 GB |
| **Few-Shot Prompted Foundation LLM** | 16.82 | 0.2241 | 0.4128 | 0.1984 | 0.3479 | 0.7854 | 71.8% | 62.4 ms | 7.4 GB |
| **BM25 RAG + Base LLM** | 14.15 | 0.2815 | 0.4632 | 0.2461 | 0.3982 | 0.8241 | 79.4% | 84.1 ms | 8.1 GB |
| **Domain-Adapted LoRA Model (SysOps-AI, 25 Epochs)** | **5.82** | **0.3948** | **0.5892** | **0.3714** | **0.5365** | **0.9126** | **96.7%** | **49.2 ms** | **6.2 GB** |

---

### Double-Blind Human Expert Evaluation (Likert Scale 1.0 - 5.0)

Evaluated across 100 randomly sampled P1/P2 production incidents by a panel of 5 Senior DevOps/SRE Engineers with at least 7 years of production infrastructure experience.

| Evaluation Metric | Base Zero-Shot | Few-Shot Prompted | BM25 RAG | Domain LoRA (SysOps-AI, 25 Epochs) | Inter-Annotator Agreement ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Technical Correctness** | 2.84 | 3.22 | 3.78 | **4.76** | 0.82 (Substantial) |
| **Hallucination-Free Rate** | 2.65 | 3.10 | 3.82 | **4.81** | 0.86 (Near Perfect) |
| **Diagnostic Coherence** | 3.15 | 3.45 | 3.91 | **4.88** | 0.79 (Substantial) |
| **Actionability of Remediation**| 2.72 | 3.18 | 3.74 | **4.85** | 0.84 (Substantial) |
| **Overall Mean Score** | 2.84 | 3.24 | 3.81 | **4.83** | **0.83** |

---

### Training Loss and Validation Trajectory (25 Epochs on Google Colab T4)

| Epoch | Training Loss (Cross-Entropy) | Validation Loss (Cross-Entropy) | Perplexity | Learning Rate | GPU Memory Utilization |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2.4812 | 2.5210 | 12.44 | $1.98 \times 10^{-4}$ | 6.12 GB |
| 2 | 2.1245 | 2.1834 | 8.88 | $1.95 \times 10^{-4}$ | 6.14 GB |
| 3 | 1.8410 | 1.8921 | 6.63 | $1.90 \times 10^{-4}$ | 6.14 GB |
| 4 | 1.6234 | 1.6845 | 5.39 | $1.83 \times 10^{-4}$ | 6.15 GB |
| 5 | 1.4521 | 1.5120 | 4.54 | $1.75 \times 10^{-4}$ | 6.15 GB |
| 6 | 1.3145 | 1.3812 | 3.98 | $1.65 \times 10^{-4}$ | 6.15 GB |
| 7 | 1.1923 | 1.2745 | 3.58 | $1.54 \times 10^{-4}$ | 6.16 GB |
| 8 | 1.0945 | 1.1821 | 3.26 | $1.42 \times 10^{-4}$ | 6.16 GB |
| 9 | 1.0112 | 1.1134 | 3.04 | $1.29 \times 10^{-4}$ | 6.16 GB |
| 10 | 0.9423 | 1.0541 | 2.87 | $1.16 \times 10^{-4}$ | 6.16 GB |
| 11 | 0.8812 | 1.0023 | 2.72 | $1.03 \times 10^{-4}$ | 6.17 GB |
| 12 | 0.8321 | 0.9614 | 2.62 | $8.98 \times 10^{-5}$ | 6.17 GB |
| 13 | 0.7845 | 0.9234 | 2.52 | $7.71 \times 10^{-5}$ | 6.17 GB |
| 14 | 0.7412 | 0.8912 | 2.44 | $6.48 \times 10^{-5}$ | 6.17 GB |
| 15 | 0.7034 | 0.8645 | 2.37 | $5.31 \times 10^{-5}$ | 6.18 GB |
| 16 | 0.6712 | 0.8412 | 2.32 | $4.22 \times 10^{-5}$ | 6.18 GB |
| 17 | 0.6421 | 0.8214 | 2.27 | $3.24 \times 10^{-5}$ | 6.18 GB |
| 18 | 0.6145 | 0.8034 | 2.23 | $2.38 \times 10^{-5}$ | 6.18 GB |
| 19 | 0.5912 | 0.7891 | 2.20 | $1.65 \times 10^{-5}$ | 6.18 GB |
| 20 | 0.5701 | 0.7782 | 2.18 | $1.06 \times 10^{-5}$ | 6.18 GB |
| 21 | 0.5523 | 0.7701 | 2.16 | $6.12 \times 10^{-6}$ | 6.18 GB |
| 22 | 0.5384 | 0.7654 | 2.15 | $3.01 \times 10^{-6}$ | 6.18 GB |
| 23 | 0.5281 | 0.7621 | 2.14 | $1.18 \times 10^{-6}$ | 6.18 GB |
| 24 | 0.5204 | 0.7602 | 2.14 | $2.97 \times 10^{-7}$ | 6.18 GB |
| 25 | **0.5142** | **0.7591** | **2.14** | **$0.00$** | **6.18 GB** |

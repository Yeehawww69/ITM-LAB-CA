
# Efficient Transformer Architectures for Low-Resource NLP

## 📌 Overview

This project compares three popular transformer models—**DistilBERT**, **ELECTRA**, and **DeBERTa**—for sentiment classification under constrained conditions:

* CPU-only training
* Limited dataset (10,000 samples)

**Goal:** Evaluate the trade-offs between accuracy and efficiency in realistic, resource-constrained deployment settings.

---

## 📊 Dataset

**IMDB Movie Reviews (Binary Sentiment Classification)**

| Split      | Samples |
| ---------- | ------- |
| Train      | 10,000  |
| Validation | 1,000   |
| Test       | 5,000   |

---

## 🤖 Models Compared

| Model      | Approach                 | Parameters |
| ---------- | ------------------------ | ---------- |
| DistilBERT | Knowledge Distillation   | 66M        |
| ELECTRA    | Replaced Token Detection | 110M       |
| DeBERTa    | Disentangled Attention   | 140M       |

---

## ⚙️ Training Setup

* **Epochs:** 3
* **Batch Size:** 32 (with gradient accumulation)
* **Learning Rate:** 2e-5
* **Max Sequence Length:** 256
* **Hardware:** CPU-only

---

## 📈 Results

### Accuracy

| Model      | Accuracy   |
| ---------- | ---------- |
| DeBERTa    | **91.24%** |
| DistilBERT | 89.28%     |
| ELECTRA    | 86.94%     |

---

### Efficiency

| Model      | Time (min) | Latency (ms) | Memory (GB) |
| ---------- | ---------- | ------------ | ----------- |
| DistilBERT | **187**    | **42**       | **3.8**     |
| ELECTRA    | 346        | 78           | 6.2         |
| DeBERTa    | 378        | 85           | 7.5         |

---

## 🔍 Key Findings

* **DeBERTa** achieves the highest accuracy
* **DistilBERT** is the most efficient (training time, memory, latency)
* **ELECTRA** performs worst under low-resource constraints

---

## 🧠 Conclusion

* Use **DeBERTa** when **accuracy is critical**
* Use **DistilBERT** for **resource-constrained environments**
* Model selection should depend on **deployment constraints**, not just benchmark scores

---

## 🛠️ Tech Stack

* PyTorch
* Hugging Face Transformers


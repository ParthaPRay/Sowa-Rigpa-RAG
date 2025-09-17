
# 🌿 Sowa-Rigpa RAG  
*An Indigenous Knowledge–Centered AI Stack for Sustainable Health Futures*

![Sowa-Rigpa](https://img.shields.io/badge/IKS-SowaRigpa-blue)  
![RAG](https://img.shields.io/badge/Framework-RAG-green)  
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview
**Sowa-Rigpa (Gso-ba Rig-pa)**, the ancient Tibetan/Himalayan medical system, balances the *three nyes-pa* (rLung – wind, mKhris-pa – bile, and Bad-kan – phlegm).  
This repository implements a **Retrieval-Augmented Generation (RAG)** framework to **digitally preserve, retrieve, and extend indigenous knowledge** in a **transparent, community-first, and offline-friendly way**.  

Core stack:
- **Docling** → Document parsing + hybrid chunking (fallback: hierarchical)  
- **Ollama** → Local, laptop-friendly LLM inference  
- **LangChain** → Embedding integration with Ollama  
- **Milvus Lite / Milvus Server** → Vector database for knowledge storage  
- **Gradio** → Interactive web-based UI  

---

## ⚡ Features
- Upload **PDF / DOCX / MD / TXT**  
- Automatic **chunking** with Docling (hybrid or hierarchical)  
- Vector embeddings with **Ollama embeddings**  
- Store/retrieve in **Milvus** (Lite or remote server)  
- **Top-K retrieval** with transparent citations  
- Runs **fully offline**, designed for **eco-conscious laptop/edge use**  
- Logging:  
  - JSONL query history  
  - CSV metrics (`ollama_metrics.csv`)  

---

## 🏗️ System Workflow

```mermaid
flowchart TD
    A[User Uploads Document / Question] --> B[Docling Parser + Chunking]
    B --> C[Ollama Embeddings]
    C --> D[Milvus Vector DB]
    D -->|Top-K Retrieval| E[Context Builder]
    E --> F[Ollama LLM - Generate Answer]
    F --> G[Answer + Metrics + Sources via Gradio UI]
````

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/sowa-rigpa-rag.git
cd sowa-rigpa-rag
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run locally

```bash
python sowa_rigpa_rag.py
```

Access the Gradio UI at:
👉 [http://localhost:7860](http://localhost:7860)

### Upload Documents

* PDF, DOCX, MD, TXT supported
* TXT bypasses Docling (direct splitter)

### Ask Questions

* Enter question → retrieves Top-K passages from Milvus
* Builds context prompt → generates answer with Ollama

---

## 📊 Logging

* **`logs/history.jsonl`** → Q\&A with contexts and metrics
* **`logs/ollama_metrics.csv`** → performance metrics (latency, tokens/sec, etc.)

---

## 🌍 Applications

* Assist **Amchi practitioners** with diagnostics & formulations
* Herb–drug interaction alerts
* Climate-aware substitution for endangered herbs
* Educational & research support for **Indian Knowledge Systems (IKS)**

---

## 🔮 Future Directions

* Expand multilingual support (Tibetan, Hindi, English)
* IoT integration for climate/ecology-aware advice
* Patient-centered health advisory tools
* Cross-domain adaptation for **Ayurveda, Siddha, Unani**

---

## 🧑‍💻 Author

**Partha Pratim Ray**
Assistant Professor, Department of Computer Applications
Sikkim University (Central University), Gangtok, India

---

## 📜 License

This project is licensed under the MIT License.

```

---

⚡ I kept it **professional + research-oriented** while making it **developer-friendly** with badges, mermaid diagram, and clear steps.  

Do you want me to also prepare a **`requirements.txt`** file (with the correct package versions: `docling`, `langchain`, `pymilvus`, `gradio`, etc.) so your repo is immediately runnable?


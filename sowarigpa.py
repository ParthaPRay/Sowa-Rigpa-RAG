#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sowa-Rigpa RAG — Docling + LangChain + Ollama + Milvus (Lite) + Gradio

- Upload PDF/DOCX/MD/TXT (TXT bypasses Docling; others via Docling)
- Docling Hybrid chunking with context-enriched serialization (fallback to Hierarchical)
- Persist vectors in Milvus Lite / Milvus Server; no re-chunking on restart (file-hash dedupe)
- GPU/CPU accelerator options for Docling; OCR toggle (off by default for speed)
- Works even with zero uploads (ships with small base Sowa-Rigpa corpus)
"""

import os
import re
import json
import time
import uuid
import hashlib
import requests
import csv
from typing import List, Tuple

import gradio as gr

# -------- Docling (conversion + chunking) ----------
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# Preferred: Hybrid chunking (tokenization-aware + hierarchical)
try:
    from docling.chunking import HybridChunker  # new chunking API
    HAS_HYBRID = True
except Exception:
    HAS_HYBRID = False

# Fallback: Hierarchical chunker
from docling_core.transforms.chunker import HierarchicalChunker

# -------- LangChain (embeddings) ----------
try:
    from langchain_ollama import OllamaEmbeddings as LC_OllamaEmbeddings
except Exception:
    # fallback if partner pkg not available
    from langchain_community.embeddings import OllamaEmbeddings as LC_OllamaEmbeddings

# -------- Milvus (Lite or Server) ----------
from pymilvus import MilvusClient

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Milvus: Lite by default (single file). Set MILVUS_URI to server for remote Milvus.
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_lite.db")
COLLECTION_BASE = os.getenv("MILVUS_COLLECTION_BASE", "sowa_rigpa_knowledge")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Embeddings & LLM
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "granite3.1-moe:latest")
LLM_CHOICES = [
    "granite3.1-moe:latest",
    "gemma3:12b", "gemma3:4b",
    "qwen3:8b", "qwen3:4b", "qwen3:1.7b",
    "llama3.2:1b",
    "deepseek-r1:8b",
    "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M",
]

# Files
BASE_CORPUS_PATH = os.path.join(DATA_DIR, "base_sowa_rigpa.txt")
REGISTRY_PATH = os.path.join(DATA_DIR, "ingested_docs.json")
HISTORY_LOG = os.path.join(LOG_DIR, "history.jsonl")
METRICS_CSV = os.path.join(LOG_DIR, "ollama_metrics.csv")

TOP_K_DEFAULT = 4
INDEX_METRIC = "COSINE"  # or "IP"

# Docling accelerator env
DOCLING_THREADS = int(os.getenv("DOCLING_THREADS", "4"))
DOCLING_OCR = os.getenv("DOCLING_OCR", "0") in ("1", "true", "True", "yes", "YES")
_device = os.getenv("DOCLING_DEVICE", "AUTO").upper()
if _device not in ("CUDA", "CPU", "AUTO"):
    _device = "AUTO"

# -----------------------------
# Utils
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sanitize_for_collection(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return out[:48].strip("_") or "default"

def _empty_registry():
    # New format: per-collection registry
    return {"by_collection": {}, "legacy_docs": {}}  # keep legacy slot for migration only

def load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                reg = json.load(f)
            # migrate old shape {"docs": {}} -> new shape
            if "docs" in reg and "by_collection" not in reg:
                new_reg = _empty_registry()
                new_reg["legacy_docs"] = reg.get("docs", {})
                return new_reg
            return reg
        except Exception:
            return _empty_registry()
    return _empty_registry()

def save_registry(reg: dict) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

def ensure_base_corpus_file():
    if not os.path.exists(BASE_CORPUS_PATH):
        with open(BASE_CORPUS_PATH, "w", encoding="utf-8") as f:
            f.write(
                "Sowa Rigpa (Gso-ba Rig-pa), the Tibetan/Himalayan medical system, "
                "balances three nyes-pa (humors): rLung (wind), mKhris-pa (bile), "
                "and Bad-kan (phlegm). Diagnosis uses interrogation, pulse, and urine "
                "analysis; therapies include diet, behavior, herbal formulas, and "
                "external therapies. Recognized by India's AYUSH; coexists with biomedicine."
            )

def friendly_err(e: Exception) -> str:
    return f"❗ {type(e).__name__}: {str(e)}"

# -----------------------------
# Embeddings + Milvus client
# -----------------------------
def get_embedder() -> LC_OllamaEmbeddings:
    return LC_OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_embedding_dim(embedder: LC_OllamaEmbeddings) -> int:
    vec = embedder.embed_query("dimension probe")
    return len(vec)

def get_collection_name(embed_model: str) -> str:
    return f"{COLLECTION_BASE}__{sanitize_for_collection(embed_model)}"

def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)

def ensure_collection(mclient: MilvusClient, collection_name: str, dim: int):
    if not mclient.has_collection(collection_name):
        # MilvusClient auto-creates fields: id (PK), vector, plus dynamic JSON for extras
        mclient.create_collection(
            collection_name=collection_name,
            dimension=dim,
            metric_type=INDEX_METRIC,
            consistency_level="Strong",
        )
    mclient.load_collection(collection_name)

# -----------------------------
# Docling converter (with accelerator)
# -----------------------------
def make_docling_converter() -> DocumentConverter:
    # Accelerator options
    dev = {
        "CUDA": AcceleratorDevice.CUDA,
        "CPU": AcceleratorDevice.CPU,
        "AUTO": AcceleratorDevice.AUTO,
    }[_device]
    acc = AcceleratorOptions(num_threads=DOCLING_THREADS, device=dev)

    # PDF pipeline options (OCR off by default for speed)
    pdf_opts = PdfPipelineOptions()
    pdf_opts.accelerator_options = acc
    pdf_opts.do_ocr = DOCLING_OCR
    pdf_opts.do_table_structure = True
    pdf_opts.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
            InputFormat.CSV,
            InputFormat.XLSX,
            InputFormat.XML_USPTO,
            InputFormat.XML_JATS,
            InputFormat.METS_GBS,
            InputFormat.JSON_DOCLING,
            InputFormat.IMAGE,
            InputFormat.AUDIO,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        },
    )

# -----------------------------
# Chunking helpers
# -----------------------------
def extract_chunks_with_docling(path: str) -> List[str]:
    """
    Use Docling + HybridChunker if available (context-enriched), otherwise Hierarchical.
    """
    conv = make_docling_converter()
    res = conv.convert(path)
    doc = res.document

    chunks: List[str] = []

    if HAS_HYBRID:
        try:
            hchunk = HybridChunker()  # default tokenizer; HF warning is benign
            for ch in hchunk.chunk(dl_doc=doc):
                enriched = hchunk.contextualize(chunk=ch)  # add headings/table/picture cues
                t = (enriched or "").strip()
                if t:
                    chunks.append(t)
            if chunks:
                return chunks
        except Exception:
            # Fall back to hierarchical below
            pass

    # Fallback: Hierarchical chunker (structure-aware paragraphs/sections)
    h = HierarchicalChunker()
    for ch in h.chunk(doc):
        t = (ch.text or "").strip()
        if t:
            chunks.append(t)
    return chunks

# LangChain splitter used ONLY for plain .txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
TXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)

def extract_chunks_any(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [t for t in TXT_SPLITTER.split_text(text) if t.strip()]
    # For all other allowed formats, go through Docling
    return extract_chunks_with_docling(path)

# -----------------------------
# Ingestion (idempotent per-collection; preserves extension; stable numeric ids)
# -----------------------------
def make_int_id(doc_id: str, idx: int) -> int:
    # 64-bit positive int from doc_id+idx
    h = hashlib.sha1(f"{doc_id}:{idx}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)

def _get_coll_docs(reg: dict, collection_name: str) -> dict:
    return reg["by_collection"].setdefault(collection_name, {"docs": {}})["docs"]

def ingest_bytes(
    mclient: MilvusClient,
    embedder: LC_OllamaEmbeddings,
    collection_name: str,
    file_bytes: bytes,
    source_name: str,
) -> Tuple[bool, str]:
    reg = load_registry()
    doc_id = sha256_bytes(file_bytes)

    # dedupe **per collection**
    coll_docs = _get_coll_docs(reg, collection_name)
    if doc_id in coll_docs:
        meta = coll_docs[doc_id]
        return False, f"✅ Already indexed in [{collection_name}]: {source_name} (chunks={meta.get('chunks', 0)})"

    # Preserve extension so Docling can detect format (fixes TXT issue)
    ext = os.path.splitext(source_name)[1].lower()
    if ext not in (".pdf", ".docx", ".md", ".html", ".pptx", ".asciidoc", ".csv", ".xlsx", ".xml", ".json", ".png", ".jpg", ".jpeg", ".txt"):
        ext = ".bin"
    tmp_path = os.path.join(DATA_DIR, f"tmp_{uuid.uuid4().hex}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        # Chunk extraction
        chunks = extract_chunks_any(tmp_path)
        if not chunks:
            return False, f"⚠️ No text content extracted from {source_name}"

        # Embed
        vectors = embedder.embed_documents(chunks)

        # Prepare rows with explicit 'id'
        data = []
        for i, (txt, vec) in enumerate(zip(chunks, vectors)):
            data.append({
                "id": make_int_id(doc_id, i),
                "vector": vec,
                "text": txt,
                "source": source_name,
                "doc_id": doc_id,
                "chunk_index": i,
            })

        # Insert
        mclient.insert(collection_name=collection_name, data=data)
        mclient.flush(collection_name)

        # Registry (per collection)
        coll_docs[doc_id] = {
            "source": source_name,
            "chunks": len(chunks),
            "ingested_at": int(time.time()),
        }
        save_registry(reg)
        return True, f"✅ Indexed {len(chunks)} chunks into [{collection_name}] from {source_name}"

    except Exception as e:
        return False, friendly_err(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _collection_is_empty(mclient: MilvusClient, collection_name: str, embedder: LC_OllamaEmbeddings) -> bool:
    """
    Heuristic: run a 1-NN search with a probe vector. If zero hits, treat as empty.
    Works for Milvus Lite and Server.
    """
    try:
        probe = embedder.embed_query("collection emptiness probe")
        mclient.load_collection(collection_name)
        res = mclient.search(
            collection_name=collection_name,
            data=[probe],
            limit=1,
            search_params={"metric_type": INDEX_METRIC, "params": {}},
            output_fields=["text"],
        )
        return not res or not res[0]
    except Exception:
        # If search fails due to no segments, treat as empty
        return True

def ensure_base_corpus(mclient: MilvusClient, embedder: LC_OllamaEmbeddings, collection_name: str):
    ensure_base_corpus_file()
    if not _collection_is_empty(mclient, collection_name, embedder):
        return
    with open(BASE_CORPUS_PATH, "rb") as f:
        b = f.read()
    ok, msg = ingest_bytes(mclient, embedder, collection_name, b, os.path.basename(BASE_CORPUS_PATH))
    print("[Bootstrap]", msg)

# -----------------------------
# Retrieval + generation
# -----------------------------
def retrieve_context(
    mclient: MilvusClient,
    embedder: LC_OllamaEmbeddings,
    collection_name: str,
    query: str,
    k: int,
) -> List[dict]:
    qvec = embedder.embed_query(query)
    mclient.load_collection(collection_name)
    res = mclient.search(
        collection_name=collection_name,
        data=[qvec],
        limit=int(k),
        search_params={"metric_type": INDEX_METRIC, "params": {}},
        output_fields=["text", "source", "doc_id", "chunk_index"],
    )
    out = []
    if res and res[0]:
        for hit in res[0]:
            ent = hit["entity"]
            out.append({
                "text": ent.get("text", ""),
                "source": ent.get("source", "unknown"),
                "doc_id": ent.get("doc_id", ""),
                "chunk_index": ent.get("chunk_index", -1),
                "score": hit.get("distance", None),
            })
    return out

def build_prompt(question: str, contexts: List[dict]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)]) or "No retrieved context."
    return (
        "You are a helpful assistant specialized in Sowa-Rigpa (Tibetan medicine). "
        "Use the provided context first. If the answer is not in the context, answer from general knowledge and say so.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def call_ollama_generate(model: str, prompt: str, temperature: float = 0.2):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()

    answer = data.get("response", "")
    total = data.get("total_duration", 0)
    load = data.get("load_duration", 0)
    pe_cnt = data.get("prompt_eval_count", 0)
    pe_dur = data.get("prompt_eval_duration", 0)
    ev_cnt = data.get("eval_count", 0)
    ev_dur = data.get("eval_duration", 0)

    tps = None
    try:
        if ev_dur and ev_cnt:
            tps = ev_cnt / (ev_dur / 1e9)
    except Exception:
        pass

    metrics = {
        "model": model,
        "total_duration_ns": total,
        "load_duration_ns": load,
        "prompt_eval_count": pe_cnt,
        "prompt_eval_duration_ns": pe_dur,
        "eval_count": ev_cnt,
        "eval_duration_ns": ev_dur,
        "tokens_per_second": tps,
    }
    return answer, metrics

def format_sources(contexts: List[dict]) -> str:
    if not contexts:
        return "No retrieved context (answered from general knowledge)."
    lines = []
    for i, c in enumerate(contexts, 1):
        src = c.get("source", "unknown")
        idx = c.get("chunk_index", -1)
        sc = c.get("score", None)
        if sc is not None:
            lines.append(f"[{i}] {src} (chunk {idx})  score={sc:.4f}")
        else:
            lines.append(f"[{i}] {src} (chunk {idx})")
    return "\n".join(lines)

# ---- CSV logging helper ----
def append_metrics_csv(
    csv_path: str,
    ts: int,
    question: str,
    answer: str,
    llm_model: str,
    embed_model: str,
    metrics: dict,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "timestamp",
        "llm_model",
        "embed_model",
        "question",
        "answer",
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "eval_count",
        "eval_duration_ns",
        "tokens_per_second",
    ]
    row = {
        "timestamp": ts,
        "llm_model": llm_model,
        "embed_model": embed_model,
        "question": question,
        "answer": answer,
        "total_duration_ns": metrics.get("total_duration_ns"),
        "load_duration_ns": metrics.get("load_duration_ns"),
        "prompt_eval_count": metrics.get("prompt_eval_count"),
        "prompt_eval_duration_ns": metrics.get("prompt_eval_duration_ns"),
        "eval_count": metrics.get("eval_count"),
        "eval_duration_ns": metrics.get("eval_duration_ns"),
        "tokens_per_second": metrics.get("tokens_per_second"),
    }
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,   # ✅ ensure all fields are safely quoted
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def answer_query(
    mclient: MilvusClient,
    embedder: LC_OllamaEmbeddings,
    collection_name: str,
    question: str,
    model: str,
    temperature: float,
    k: int,
) -> Tuple[str, str, str]:
    if not question or not question.strip():
        return "Please enter a question.", "{}", "—"

    contexts = retrieve_context(mclient, embedder, collection_name, question.strip(), k)
    prompt = build_prompt(question, contexts)
    try:
        answer, metrics = call_ollama_generate(model=model, prompt=prompt, temperature=temperature)
    except Exception as e:
        return friendly_err(e), "{}", format_sources(contexts)

    ts = int(time.time())

    # JSONL history
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": ts,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "metrics": metrics,
        }) + "\n")

    # CSV metrics append
    try:
        append_metrics_csv(
            csv_path=METRICS_CSV,
            ts=ts,
            question=question,
            answer=answer,
            llm_model=model,
            embed_model=EMBED_MODEL,
            metrics=metrics,
        )
    except Exception as e:
        print(f"[CSV logging error] {friendly_err(e)}")

    return answer, json.dumps(metrics, indent=2), format_sources(contexts)

# -----------------------------
# Gradio UI
# -----------------------------
def build_ui(mclient: MilvusClient, embedder: LC_OllamaEmbeddings, collection_name: str):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🌿 Sowa-Rigpa RAG — Docling Hybrid Chunking + Ollama + Milvus")

        with gr.Tab("Ask"):
            q = gr.Textbox(label="Ask a Sowa-Rigpa question", placeholder="e.g., What are the three nyes-pa (humors)?")
            with gr.Row():
                model = gr.Dropdown(choices=LLM_CHOICES, value=DEFAULT_LLM, label="Ollama LLM")
                temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                k = gr.Slider(1, 10, value=TOP_K_DEFAULT, step=1, label="Top-K retrieved")
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=10)
            metrics = gr.Textbox(label="Ollama Metrics (JSON)", lines=10)
            sources = gr.Textbox(label="Retrieved Sources")

            ask_btn.click(
                lambda question, model, temp, k: answer_query(
                    mclient, embedder, collection_name, question, model, temp, int(k)
                ),
                inputs=[q, model, temp, k],
                outputs=[answer, metrics, sources],
            )

        with gr.Tab("Manage Knowledge"):
            gr.Markdown(
                "Upload **PDF / DOCX / MD / TXT**. Files are **deduplicated** per embedding collection; "
                "no re-chunking on restart. PDFs use GPU if available (see env vars)."
            )
            files = gr.Files(label="Upload documents", file_types=[".pdf", ".docx", ".md", ".txt"], type="filepath")
            ingest_btn = gr.Button("Ingest")
            ingest_log = gr.Textbox(label="Ingestion Log", lines=12)

            def do_ingest(paths: List[str]) -> str:
                if not paths:
                    return "No files selected."
                logs = []
                for p in paths:
                    try:
                        with open(p, "rb") as f:
                            b = f.read()
                        ok, msg = ingest_bytes(mclient, embedder, collection_name, b, os.path.basename(p))
                        logs.append(msg)
                    except Exception as e:
                        logs.append(friendly_err(e))
                return "\n".join(logs)

            ingest_btn.click(do_ingest, inputs=[files], outputs=[ingest_log])

        gr.Markdown(
            "Tip: Pull models in Ollama first (e.g., `granite-embedding:278m`, `granite3.1-moe:latest`). "
            "Set `DOCLING_DEVICE`, `DOCLING_THREADS`, `DOCLING_OCR` to tune speed/accuracy."
        )
    return demo

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        embedder = get_embedder()
        dim = get_embedding_dim(embedder)
        collection_name = get_collection_name(EMBED_MODEL)

        mclient = get_milvus_client()
        ensure_collection(mclient, collection_name, dim)
        ensure_base_corpus(mclient, embedder, collection_name)
    except Exception as e:
        print(f"[Startup error] {friendly_err(e)}")
        raise

    app = build_ui(mclient, embedder, collection_name)
    app.launch(server_name="0.0.0.0", server_port=7860)

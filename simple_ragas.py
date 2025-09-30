# src/simple_ragas.py  (public / NDA-safe)
import os
import sys
import logging
import traceback as tb
from typing import List, Optional

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------
# Config / constants
# -----------------------------
load_dotenv()

_THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(_THIS_FILE)

DEFAULT_DATA_FOLDER = "data"  # was "Pro_Election"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
FAISS_DIR = os.path.join(PROJECT_ROOT, "faiss_index")

LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "huggingface"

GROQ_MODEL_ID = "llama-3.3-70b-versatile"
HUGGINGFACE_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Paths / logging
# -----------------------------
def _path_in_data(*parts: str) -> str:
    base = os.path.join(PROJECT_ROOT, DEFAULT_DATA_FOLDER)
    return os.path.join(base, *parts)

def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(_path_in_data(), exist_ok=True)

def get_logger(filename: str):
    _ensure_dirs()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(LOG_FILE, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    return logging.getLogger(os.path.basename(filename))

# -----------------------------
# Prompt (concise, safe)
# -----------------------------
PROMPT = PromptTemplate(
    template=(
        "You are an assistant answering questions using only the provided context. "
        "Cite facts concisely and be direct. If the answer is not in the context, say you don't know.\n\n"
        "<context>\n{context}\n</context>\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

# -----------------------------
# Load documents
# -----------------------------
def load_all_pdfs_recursively() -> List[Document]:
    logger = get_logger(__file__)
    try:
        base_dir = _path_in_data()
        if not os.path.isdir(base_dir):
            logger.error(f"Data folder not found: {base_dir}")
            return []

        docs: List[Document] = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    fpath = os.path.join(root, fname)
                    try:
                        docs.extend(PyPDFLoader(fpath).load())
                    except Exception:
                        logger.warning(f"Failed to load PDF: {fpath} - {tb.format_exc()}")

        if not docs:
            logger.error(f"No PDF documents found recursively under {DEFAULT_DATA_FOLDER}/")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        logger.info("Splitting documents (recursive)")
        return splitter.split_documents(docs)
    except Exception:
        logger.error(f"Error occurred while loading PDFs - {tb.format_exc()}")
        return []

def load_pdfs_from_subfolders(subfolders: List[str]) -> List[Document]:
    logger = get_logger(__file__)
    try:
        docs: List[Document] = []
        for sf in subfolders:
            folder = _path_in_data(sf)
            if not os.path.isdir(folder):
                logger.warning(f"Skipping missing subfolder: {folder}")
                continue
            for root, _, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith(".pdf"):
                        fpath = os.path.join(root, fname)
                        try:
                            docs.extend(PyPDFLoader(fpath).load())
                        except Exception:
                            logger.warning(f"Failed to load PDF: {fpath} - {tb.format_exc()}")

        if not docs:
            logger.error("No PDF documents found in the selected subfolder(s).")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        logger.info("Splitting documents (selected folders)")
        return splitter.split_documents(docs)
    except Exception:
        logger.error(f"Error occurred while loading from selected folders - {tb.format_exc()}")
        return []

def load_data(all_folders: bool = True, subfolders: Optional[List[str]] = None) -> List[Document]:
    """Preserves your existing signature (used by app.py)."""
    logger = get_logger(__file__)
    try:
        if all_folders:
            return load_all_pdfs_recursively()
        if subfolders:
            return load_pdfs_from_subfolders(subfolders)
        return load_pdfs_from_subfolders(["."])
    except Exception:
        logger.error(f"Error in load_data() - {tb.format_exc()}")
        return []

# -----------------------------
# Embeddings / LLM
# -----------------------------
def get_huggingface_embeddings():
    logger = get_logger(__file__)
    try:
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL_ID,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        logger.error(f"Error creating Hugging Face embeddings - {tb.format_exc()}")
        return None

def get_embeddings_by_provider():
    if EMBEDDING_PROVIDER.lower() == "huggingface":
        return get_huggingface_embeddings()
    get_logger(__file__).error(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
    return None

def get_groq_llm():
    logger = get_logger(__file__)
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("Required API key not found in environment")
            return None
        return ChatGroq(
            model=GROQ_MODEL_ID,
            temperature=0.1,
            max_tokens=512,
            stop_sequences=None,
        )
    except Exception:
        logger.error(f"Error creating Groq model - {tb.format_exc()}")
        return None

def get_llm_by_provider():
    if LLM_PROVIDER.lower() == "groq":
        return get_groq_llm()
    get_logger(__file__).error(f"Unknown LLM provider: {LLM_PROVIDER}")
    return None

# -----------------------------
# Vector store
# -----------------------------
def create_vector_store(docs: List[Document]):
    logger = get_logger(__file__)
    try:
        embeddings = get_embeddings_by_provider()
        if embeddings is None:
            logger.error("Embeddings instance is None. Cannot create vector store.")
            return None
        logger.info("Creating FAISS vector store")
        return FAISS.from_documents(documents=docs, embedding=embeddings)
    except Exception:
        logger.error(f"Error creating vector store - {tb.format_exc()}")
        return None

def save_vector_store(vector_store):
    logger = get_logger(__file__)
    try:
        vector_store.save_local(FAISS_DIR)
        logger.info("Saved vector store to disk")
    except Exception:
        logger.error(f"Error saving vector store - {tb.format_exc()}")

def load_vector_store():
    logger = get_logger(__file__)
    try:
        embeddings = get_embeddings_by_provider()
        if embeddings is None:
            logger.error("Embeddings instance is None. Cannot load vector store.")
            return None
        if not os.path.isdir(FAISS_DIR):
            logger.error(f"FAISS folder not found: {FAISS_DIR}")
            return None
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        logger.error(f"Error loading vector store - {tb.format_exc()}")
        return None

# -----------------------------
# QA chain
# -----------------------------
def get_response(llm, vector_store_faiss, query: str):
    """
    Returns (answer, contexts) where contexts are strings tagged with short source hints.
    """
    logger = get_logger(__file__)
    try:
        retriever = vector_store_faiss.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.5},
        )
        retrieved_docs = retriever.get_relevant_documents(query)

        def ctx_str(d):
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_path") or ""
            page = meta.get("page", "")
            tag = f"[{os.path.basename(src)}#p{page}]" if src else ""
            return f"{tag}\n{d.page_content}"

        contexts_texts = [ctx_str(d) for d in retrieved_docs]

        chain = create_stuff_documents_chain(llm, PROMPT)
        qa_chain = create_retrieval_chain(retriever, chain)
        result = qa_chain.invoke({"input": query, "question": query})
        answer = (result.get("answer") or "").strip()
        return answer, contexts_texts
    except Exception:
        logger.error(f"Error generating response - {tb.format_exc()}")
        return "", []

# -----------------------------
# RAGAS evaluation (optional)
# -----------------------------
def evaluate_with_ragas_batch(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    references: Optional[List[str]] = None,
):
    """
    Batch evaluation with RAGAS. If `references` provided (gold answers),
    metrics that require gold are also computed.
    """
    try:
        rows = []
        for i in range(len(questions)):
            row = {
                "question": questions[i],
                "answer": answers[i],
                "contexts": contexts_list[i],
            }
            if references is not None and i < len(references) and references[i] and references[i].strip():
                row["reference"] = references[i].strip()
            rows.append(row)

        ds = Dataset.from_list(rows)
        has_ref = any("reference" in r for r in rows)

        mets = [faithfulness, answer_relevancy]
        if has_ref:
            mets += [context_precision, context_recall]

        result = evaluate(ds, metrics=mets)
        df = result.to_pandas()

        numeric_cols = [c for c in df.columns if c not in ("question", "answer", "contexts", "reference")]
        if numeric_cols:
            print("\n=== RAGAS macro-averages ===")
            print(df[numeric_cols].mean().to_frame("mean").T)

        return df
    except Exception as e:
        print(f"[RAGAS] evaluation failed: {e}")
        return None

# -----------------------------
# Main (local demo only)
# -----------------------------
def main():
    # Phase 1: index
    docs = load_data(all_folders=True)  # scans data/ recursively
    if not docs:
        raise Exception("No documents loaded. Add PDFs under /data and try again.")
    vectorstore_created = create_vector_store(docs=docs)
    if not vectorstore_created:
        raise Exception("Failed to create vectorstore.")
    save_vector_store(vectorstore_created)
    print("FAISS index created and saved.")

    # Phase 2: load store + LLM
    vector_store = load_vector_store()
    if not vector_store:
        raise Exception("Failed to load FAISS vector store")
    llm = get_llm_by_provider()
    if not llm:
        raise Exception(f"Failed to load LLM: {LLM_PROVIDER}")
    print(f"Using {LLM_PROVIDER} LLM with model: {GROQ_MODEL_ID}")

    # Phase 3: single query (generic demo)
    query = "Summarize the key policy notices in the documents."
    print("Query:\n", query)
    answer, contexts = get_response(llm, vector_store, query)
    if not answer:
        raise Exception("LLM response generation failed.")
    formatted = "\n".join(line.strip() for line in answer.splitlines() if line.strip())
    print("Response:\n", formatted)
    print("\nTop contexts used:")
    for i, c in enumerate(contexts, 1):
        preview = c[:600] + ("..." if len(c) > 600 else "")
        print(f"\n--- Context {i} ---\n{preview}")

    # Phase 4: RAGAS batch evaluation (generic placeholders)
    eval_questions = [
        "Summarize the key policy notices in the documents.",
        "What deadlines or timelines are mentioned?",
    ]
    eval_answers: List[str] = []
    eval_contexts: List[List[str]] = []
    for q in eval_questions:
        a, ctxs = get_response(llm, vector_store, q)
        eval_answers.append(a)
        eval_contexts.append(ctxs)

    gold_refs: Optional[List[str]] = None
    ragas_df = evaluate_with_ragas_batch(
        questions=eval_questions,
        answers=eval_answers,
        contexts_list=eval_contexts,
        references=gold_refs,
    )
    if ragas_df is not None:
        print("\n=== RAGAS per-sample scores ===")
        print(ragas_df)

if __name__ == "__main__":
    main()

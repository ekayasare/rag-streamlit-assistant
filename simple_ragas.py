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

#  Config 
load_dotenv()

CONSTANTS_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT_DIRECTORY = os.path.dirname(CONSTANTS_FILE_PATH)

DEFAULT_DATA_FOLDER = "Pro_Election"
APP_LOG_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "log")
APP_LOG_FILE_PATH = os.path.join(APP_LOG_DIRECTORY, "app.log")
VECTOR_STORE_FAISS_INDEX = os.path.join(PROJECT_ROOT_DIRECTORY, "faiss_index")

LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "huggingface"

GROQ_MODEL_ID = "llama-3.3-70b-versatile"
HUGGINGFACE_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#  Paths / Logging 
def path_in_data(*parts: str) -> str:
    base = os.path.join(PROJECT_ROOT_DIRECTORY, DEFAULT_DATA_FOLDER)
    return os.path.join(base, *parts)

def ensure_dirs():
    os.makedirs(APP_LOG_DIRECTORY, exist_ok=True)
    os.makedirs(VECTOR_STORE_FAISS_INDEX, exist_ok=True)
    os.makedirs(path_in_data(), exist_ok=True)

def get_logger(filename: str):
    ensure_dirs()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(APP_LOG_FILE_PATH, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ]
        )
    return logging.getLogger(os.path.basename(filename))

#  Prompt 
PROMPT = PromptTemplate(
    template=(
        "Human: use the following pieces of context to provide a concise answer to the question at the end, "
        "but use at least 150 words with detailed explanation. If you don't know the answer, say you don't know.\n"
        "<context>\n{context}\n</context>\n"
        "Question: {question}\n\n"
        "Assistant:"
    ),
    input_variables=["context", "question"]
)

#  Load documents 
def load_all_pdfs_recursively() -> List[Document]:
    logger = get_logger(__file__)
    try:
        base_dir = path_in_data()
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
        logger.error(f"Error occurred while loading all PDFs recursively - {tb.format_exc()}")
        return []

def load_pdfs_from_subfolders(subfolders: List[str]) -> List[Document]:
    logger = get_logger(__file__)
    try:
        docs: List[Document] = []
        for sf in subfolders:
            folder = path_in_data(sf)
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

#  Embeddings / LLM 
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
    logger = get_logger(__file__)
    if EMBEDDING_PROVIDER.lower() == "huggingface":
        return get_huggingface_embeddings()
    logger.error(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
    return None

def get_groq_llm():
    logger = get_logger(__file__)
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found in environment variables")
            return None
        return ChatGroq(
            model=GROQ_MODEL_ID,
            temperature=0.1,
            max_tokens=512,
            stop_sequences=None
        )
    except Exception:
        logger.error(f"Error creating Groq model - {tb.format_exc()}")
        return None

def get_llm_by_provider():
    logger = get_logger(__file__)
    if LLM_PROVIDER.lower() == "groq":
        return get_groq_llm()
    logger.error(f"Unknown LLM provider: {LLM_PROVIDER}")
    return None

#  Vector store 
def create_vector_store(docs: List[Document]):
    logger = get_logger(__file__)
    try:
        logger.info("Creating vector store documents")
        embeddings = get_embeddings_by_provider()
        if embeddings is None:
            logger.error("Embeddings instance is None. Cannot create vector store.")
            return None
        return FAISS.from_documents(documents=docs, embedding=embeddings)
    except Exception:
        logger.error(f"Error creating vector store - {tb.format_exc()}")
        return None

def save_vector_store(vector_store):
    logger = get_logger(__file__)
    try:
        vector_store.save_local(VECTOR_STORE_FAISS_INDEX)
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
        if not os.path.isdir(VECTOR_STORE_FAISS_INDEX):
            logger.error(f"FAISS folder not found: {VECTOR_STORE_FAISS_INDEX}")
            return None
        return FAISS.load_local(
            VECTOR_STORE_FAISS_INDEX,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        logger.error(f"Error loading vector store - {tb.format_exc()}")
        return None

#  QA chain 
def get_response(llm, vector_store_faiss, query: str):
    """
    Returns (answer, contexts) where contexts are strings tagged with short source hints.
    """
    logger = get_logger(__file__)
    try:
        retriever = vector_store_faiss.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.5}
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

#  RAGAS evaluation 
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

#  Main 
def main():
    # Phase 1: index
    docs = load_data(all_folders=True)  # scans Pro_Election/ recursively
    if not docs:
        raise Exception("No documents loaded. Check your data folder.")
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

    # Phase 3: single query (demo)
    query = "What are some actions without a meeting according to Polk Bylaws?"
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

    # Phase 4: RAGAS batch evaluation (example with 2 questions)
    eval_questions = [
        "What are some actions without a meeting according to Polk Bylaws?",
        "When are HOA board election notices due?",
    ]
    eval_answers: List[str] = []
    eval_contexts: List[List[str]] = []
    for q in eval_questions:
        a, ctxs = get_response(llm, vector_store, q)
        eval_answers.append(a)
        eval_contexts.append(ctxs)

    # If you have gold answers, put them here; else set to None
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
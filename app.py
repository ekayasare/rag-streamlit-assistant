import os
from PIL import Image
import streamlit as st

# -------------------------
# PAGE CONFIG (generic)
# -------------------------
st.set_page_config(page_title="Document Q&A (Demo)", layout="wide")

# -------------------------
# CUSTOM STYLING (neutral)
# -------------------------
st.markdown("""
<style>
    body { background-color: #f5f5dc; }
    .response-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# OPTIONAL LOGO (neutral)
# -------------------------
logo_path = os.path.join("assets", "placeholder_logo.png")
if os.path.exists(logo_path):
    st.image(Image.open(logo_path), width=80)

# -------------------------
# APP HEADER (generic)
# -------------------------
st.markdown("## Document Q&A Assistant (Demo)")
st.markdown("*Ask questions and get answers from your documents.*")

# -------------------------
# FOLDER PATHS (consistent with README)
# -------------------------
UPLOADS_FOLDER = "uploads_for_index"
DATA_FOLDER = "data"
INDEX_DIR = "faiss_index"

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.markdown("## Document Options")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (not stored in repo)", type="pdf", accept_multiple_files=True
)

# Save uploaded files at runtime
if uploaded_files:
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join(UPLOADS_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")

use_all_folders = st.sidebar.checkbox("Use all folders in /data", value=True)

folder_options = []
if os.path.exists(DATA_FOLDER):
    folder_options = [
        name for name in os.listdir(DATA_FOLDER)
        if os.path.isdir(os.path.join(DATA_FOLDER, name))
    ]

selected_folders = []
if not use_all_folders and folder_options:
    selected_folders = st.sidebar.multiselect(
        "Select subfolders in /data to include:",
        folder_options,
        default=folder_options[:1]
    )

rebuild_requested = st.sidebar.button("Rebuild index")

# -------------------------
# RAG HELPERS (your public module)
# -------------------------
from simple_ragas import (
    load_data,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_llm_by_provider,
    get_response
)

# -------------------------
# VECTOR STORE (build or load)
# -------------------------
index_exists = os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR)
needs_rebuild = bool(uploaded_files or not index_exists or rebuild_requested)

docs = []
vector_store = None

if needs_rebuild:
    with st.spinner("Processing documents..."):
        docs = load_data(
            all_folders=use_all_folders,
            subfolders=selected_folders if not use_all_folders else None,
            data_dir=DATA_FOLDER,
            uploads_dir=UPLOADS_FOLDER
        )
        if not docs:
            st.error("No documents found. Add PDFs to /data or upload via the sidebar.")
        else:
            vector_store = create_vector_store(docs, index_dir=INDEX_DIR)
            if vector_store:
                save_vector_store(vector_store, index_dir=INDEX_DIR)
                st.success("Vector store built and saved.")
            else:
                st.error("Vector store creation failed.")
else:
    with st.spinner("Loading existing vector store..."):
        vector_store = load_vector_store(index_dir=INDEX_DIR)
        if not vector_store:
            st.error("No index found. Please add documents and rebuild.")
            st.stop()

# -------------------------
# LLM INSTANCE (env-based)
# -------------------------
llm = get_llm_by_provider()
if not llm:
    st.error("Failed to initialize the language model. Check your environment variables.")
    st.stop()

# -------------------------
# MAIN INTERFACE
# -------------------------
st.markdown("---")
st.markdown("### Ask a question")

user_question = st.text_area("Your question", height=100)
submit_clicked = st.button("Ask")

if submit_clicked:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, contexts = get_response(llm, vector_store, user_question)

        st.markdown("### Response")
        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

        st.markdown("### Contexts used")
        if contexts:
            for i, ctx in enumerate(contexts):
                with st.expander(f"Context {i+1}"):
                    st.markdown(ctx)
        else:
            st.info("No contexts returned for this query.")

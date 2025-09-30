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
logo_path = os.path.join("assets", "placeholder_logo.png")  # was assets/logo.png
if os.path.exists(logo_path):
    st.image(Image.open(logo_path), width=80)

# -------------------------
# APP HEADER (generic)
# -------------------------
st.markdown("## Document Q&A Assistant (Demo)")
st.markdown("*Ask questions and get answers from your documents.*")

# -------------------------
# FOLDER PATHS (NDA-safe)
# -------------------------
UPLOADS_FOLDER = "uploads_for_index"
DATA_FOLDER = "data"                 # was Pro_Election
FAISS_INDEX_PATH = "faiss_index"

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.markdown("## Document Options")

uploaded_files = st.sidebar.file_uploader(
    "Upload additional PDFs", type="pdf", accept_multiple_files=True
)

# Save uploaded files
if uploaded_files:
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOADS_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

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
        "Select folders to include:",
        folder_options,
        default=folder_options[:1] if folder_options else []
    )

rebuild_index = st.sidebar.button("Rebuild index")

# -------------------------
# VECTOR STORE LOGIC
# -------------------------
from simple_ragas import (
    load_data,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_llm_by_provider,
    get_response
)

faiss_exists = os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH)
needs_rebuild = bool(uploaded_files or not faiss_exists or rebuild_index)

docs = []
vector_store = None  # prevent 'possibly unbound' errors

if needs_rebuild:
    with st.spinner("Processing documents..."):
        if use_all_folders:
            docs = load_data(all_folders=True)
        else:
            docs = load_data(all_folders=False, subfolders=selected_folders)

        if not docs:
            st.error("No documents found to process. Add PDFs to /data or upload via the sidebar.")
        else:
            vector_store = create_vector_store(docs)
            if vector_store:
                save_vector_store(vector_store)
                st.success("Vector store built and saved.")
            else:
                st.error("Vector store creation failed.")
else:
    with st.spinner("Loading existing vector store..."):
        vector_store = load_vector_store()
        if not vector_store:
            st.error("Failed to load an existing index. Please add documents and rebuild.")
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

        # Response section
        st.markdown("### Response")
        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

        # Contexts
        st.markdown("### Contexts used")
        if contexts:
            for i, ctx in enumerate(contexts):
                with st.expander(f"Context {i+1}"):
                    st.markdown(ctx)
        else:
            st.info("No contexts returned for this query.")

import streamlit as st
from PIL import Image
import os
import shutil

#  PAGE CONFIG 
st.set_page_config(page_title="HOA Document Q&A", layout="wide")

#  CUSTOM STYLING 
st.markdown("""
    <style>
        /* Global background */
        body {
            background-color: #f5f5dc;
        }
        /* Main response box */
        .response-box {
            background-color: #f0f0f0;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* Hide Streamlit footer & hamburger menu */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

#  LOGO 
logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=80)

#  APP HEADER 
st.markdown("## HOA Document Q&A Assistant")
st.markdown("*Ask questions and get answers from your community documents.*")

#  FOLDER PATHS 
UPLOADS_FOLDER = os.path.join("uploads_for_index")
DATA_FOLDER = os.path.join("Pro_Election")

#  SIDEBAR 
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

use_all_folders = st.sidebar.checkbox("Use all folders", value=True)

folder_options = []
if os.path.exists(DATA_FOLDER):
    folder_options = [
        name for name in os.listdir(DATA_FOLDER)
        if os.path.isdir(os.path.join(DATA_FOLDER, name))
    ]

selected_folders = []
if not use_all_folders:
    selected_folders = st.sidebar.multiselect(
        "Select folders to include:",
        folder_options,
        default=folder_options[:1] if folder_options else []
    )

#  VECTOR STORE LOGIC 
from simple_ragas import (
    load_data,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_llm_by_provider,
    get_response
)

FAISS_INDEX_PATH = os.path.join("faiss_index")
faiss_exists = os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH)
needs_rebuild = bool(uploaded_files or not faiss_exists)

docs = []
vector_store = None # Prevent 'possibly unbound' error
if needs_rebuild:
    with st.spinner("Processing documents..."):
        if use_all_folders:
            docs = load_data(all_folders=True)
        else:
            docs = load_data(all_folders=False, subfolders=selected_folders)

        if not docs:
            st.error("No documents found to process. Please check your folders or uploads.")
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
            st.error("Failed to load FAISS index. Please upload documents first.")
            st.stop()

#  LLM INSTANCE 
llm = get_llm_by_provider()
if not llm:
    st.error("Failed to load LLM. Check your GROQ_API_KEY.")
    st.stop()

#  MAIN INTERFACE 
st.markdown("---")
st.markdown("### Ask a question")

user_question = st.text_area("Ask a question", height=100)
submit_clicked = st.button("Ask a question")

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
        for i, ctx in enumerate(contexts):
            st.expander(f"Context {i+1}").markdown(ctx)
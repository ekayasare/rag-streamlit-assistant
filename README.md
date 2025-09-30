# RAG Streamlit Assistant

This repository contains a Retrieval-Augmented Generation (RAG) demo that allows users to query documents via a simple Streamlit interface. The system combines document preprocessing, vector search, and LLM integration to provide accurate, context-aware responses.

---

## Overview

The Document Q&A Assistant:
- Loads and indexes PDF documents
- Supports uploading additional files during runtime
- Allows natural-language questions against the document set
- Returns concise answers with supporting context
- Uses vector embeddings + similarity search for efficiency

---

## Core Technologies

- **Data Processing**: LangChain for document loading and chunking  
- **Similarity Search**: FAISS for semantic retrieval  
- **Embeddings**: HuggingFace MiniLM–L6-v2 for vector representations  
- **Generative AI**: Groq API powering Llama-3.3-70B responses  
- **Interface**: Streamlit for an intuitive web app front-end  

---

## Repository Structure

rag-streamlit-assistant/
│
├── app.py # Streamlit UI
├── simple_ragas.py # Backend pipeline logic
├── assets/
│ └── placeholder_logo.png # Optional branding image
├── data/ # Local PDFs (empty; do not commit real docs)
├── uploads_for_index/ # Runtime uploads (gitignored)
├── faiss_index/ # Vector store (gitignored)
├── .env.example # Example environment file
├── requirements.txt # Dependencies
└── README.md


---

## Getting Started

1. Clone the repository:
git clone https://github.com/ekayasare/rag-streamlit-assistant.git
cd rag-streamlit-assistant

pip install -r requirements.txt

GROQ_API_KEY=your_key_here

streamlit run app.py

## Design Highlights

- Clean, user-friendly interface with minimal technical jargon  
- Automatic vector store rebuilds only when needed  
- Transparent source display for trustworthiness  
- Lightweight UI with a subtle beige theme and structured response box  

---

## Author

**Ekay (Emmanuel Nyarko Asare)**  
AI/ML Data Scientist | Builder of AI Assistants  
[LinkedIn](https://www.linkedin.com/in/emmanuel-asare-6b952827b/) | [GitHub](https://github.com/ekayasare)

---

## NDA & Safety

This repository is a **generic demo**. All client-specific data, documents, and branding have been removed or replaced with synthetic examples in accordance with NDA requirements.

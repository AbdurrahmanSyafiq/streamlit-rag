import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from langchain_groq import ChatGroq

# =========================
# STEP 0: LOAD ENVIRONMENT
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-chunks")

if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Key atau konfigurasi belum lengkap di file .env.")
    st.stop()

# =========================
# STEP 1: CONNECT TO PINECONE (v2 style)
# =========================
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    st.sidebar.success("✅ Terhubung ke Pinecone!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal terhubung ke Pinecone: {e}")
    st.stop()

# =========================
# STEP 2: SETTINGS
# =========================
pkl_path = "parsed_documents.pkl"
window_size = 3
similarity_top_k = 80
rerank_top_n = 4

# =========================
# SIDEBAR CONTROL
# =========================
st.sidebar.title("🔧 Pengaturan")
rerank_enabled = st.sidebar.checkbox("Gunakan Reranking", value=True)

# =========================
# LOAD DOCUMENTS FROM PKL
# =========================
@st.cache_resource
def load_documents_from_pickle():
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# =========================
# BUILD QUERY ENGINE
# =========================
@st.cache_resource
def build_query_engine(_documents):
    embed_model = HuggingFaceEmbedding(model_name="indobenchmark/indobert-large-p1")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=0,
        model_name="llama-3.3-70b-versatile"
    )

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="default"  # optional, namespace di Pinecone
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index_obj = VectorStoreIndex.from_documents(
        documents=_documents,
        embed_model=embed_model,
        storage_context=storage_context,
        llm=llm,
        node_parser=node_parser,
    )

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-large")

    post_proc = [postproc, rerank] if rerank_enabled else [postproc]

    return index_obj.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=post_proc,
        llm=llm
    )

# =========================
# MAIN UI
# =========================
st.title("📄 QA Dokumen PPNPK Retinoblastoma (via PKL + Pinecone v2)")

query = st.text_area("💬 Pertanyaan", placeholder="Contoh: Apa saja indikasi medikasi Sevofluran?", height=100)

if query:
    with st.spinner("⏳ Mencari jawaban..."):
        documents = load_documents_from_pickle()
        query_engine = build_query_engine(documents)
        response = query_engine.query(query)

        st.subheader("📌 Jawaban")
        st.markdown(response.response)

        st.subheader("📚 Konteks")
        for node in response.source_nodes:
            st.markdown(f"- {node.node.text.strip()}")

import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
import weaviate
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
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
if not GROQ_API_KEY:
    st.error("❌ GROQ API Key tidak ditemukan! Pastikan sudah menyimpan API Key di file .env.")
    st.stop()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "DocumentChunks")

# =========================
# STEP 1: CONNECT TO WEAVIATE
# =========================
try:
    weaviate_client = weaviate.connect_to_local()
    st.sidebar.success("✅ Terhubung ke Weaviate!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal terhubung ke Weaviate: {e}")
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

    vector_store = WeaviateVectorStore(
        weaviate_client=weaviate_client,
        index_name=WEAVIATE_COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=_documents,
        embed_model=embed_model,
        storage_context=storage_context,
        llm=llm,
        node_parser=node_parser,
    )

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-large")

    post_proc = [postproc, rerank] if rerank_enabled else [postproc]

    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=post_proc,
        llm=llm
    )

# =========================
# MAIN UI
# =========================
st.title("📄 QA Dokumen PPNPK Retinoblastoma (via PKL)")

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

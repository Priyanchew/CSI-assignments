import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.llms import HuggingFaceHub
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor


load_dotenv() 

@st.cache_resource(show_spinner="Loading BGE encoder & Llama-3 …")
def load_models():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    return embeddings, llm

embeddings, hf_llm = load_models()

@st.cache_resource(show_spinner="Indexing Ollama blog …")
def build_web_retriever():
    loader = WebBaseLoader("https://ollama.com/blog/embedding-models")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb.as_retriever()

static_web_retriever = build_web_retriever()

st.title("📄🔍 LangChain RAG Playground")

with st.sidebar:
    st.header("Settings")
    use_web = st.checkbox("Use Web Search (Wikipedia + blog)", value=True)
    uploaded_pdf = st.file_uploader(
        "Upload a PDF", type=["pdf"],
        help="Optional – embed & query your own document"
    )
    st.markdown("---")
    query = st.text_input("Ask a question and press **Enter**:", "")

def get_tools():
    tools = []

    if use_web:
        wiki_api = WikipediaAPIWrapper(top_k_results=1,
                                       doc_content_chars_max=200)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
        tools.append(wiki_tool)

        web_tool = create_retriever_tool(
            static_web_retriever,
            name="Ollama Embedding Model",
            description="Search for info about Ollama embedding models."
        )
        tools.append(web_tool)

    if uploaded_pdf:
        if "pdf_retriever" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".pdf") as tf:
                tf.write(uploaded_pdf.read())
                pdf_path = tf.name
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            pdf_chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(pages)
            pdf_db = FAISS.from_documents(pdf_chunks, embeddings)
            st.session_state["pdf_retriever"] = pdf_db.as_retriever()

        pdf_tool = create_retriever_tool(
            st.session_state["pdf_retriever"],
            name="User PDF",
            description="Search the uploaded PDF."
        )
        tools.append(pdf_tool)

    return tools

if query:
    tools = get_tools()
    if not tools:
        st.warning("Please enable Web Search or upload a PDF to provide context.")
        st.stop()

    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(hf_llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.spinner("Thinking…"):
        response = executor.invoke({"input": query})["output"]

    st.markdown("#### Answer")
    st.write(response)
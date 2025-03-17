import streamlit as st
import chromadb
import requests
from io import BytesIO
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings

st.set_page_config(page_title="DocuMind AI", layout="wide")

st.sidebar.title("⚙️ Model Selection")
selected_model = st.sidebar.radio(
    "Choose an AI Model:",
    ["deepseek-r1:1.5b", "llama3.2:3b", "llama3.2:1b"]
)

PROMPT_TEMPLATE = """
You are an expert research gprMax assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'documents/'
CHROMA_DB_PATH = "chroma_db"
LANGUAGE_MODEL = OllamaLLM(model=selected_model)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = OllamaEmbeddings(model=selected_model)

collection = chroma_client.get_or_create_collection(
    name="document_vectors",
    metadata={"hnsw:space": "cosine"} 
)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def download_file_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split("/")[-1]   
            file_path = PDF_STORAGE_PATH + file_name
            with open(file_path, "wb") as file:
                file.write(response.content)
            return file_path
        else:
            st.error("Failed to download file. Please check the URL.")
            return None
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_text):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_text(raw_text)

def index_documents(document_chunks):
    for idx, doc in enumerate(document_chunks):
        collection.add(
            ids=[f"doc_{idx}"],
            documents=[doc],
            metadatas=[{"source": "uploaded_file"}],
        )

def find_related_documents(query):
    results = collection.query(query_texts=[query], n_results=3)

    if results and "documents" in results and results["documents"]:
        return results["documents"][0]   
    
    return []   

def generate_answer(user_query, context_documents):
    if not context_documents:
        return "I couldn't find relevant information in the document."

    context_text = "\n\n".join(context_documents)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

st.title("gprMax Chatbot")
st.markdown("### Ask Questions related to gprMax")
st.markdown("---")

st.subheader("📂 Upload a PDF, TXT, or RST File or Enter a File URL")
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "rst"], accept_multiple_files=False)
file_url = st.text_input("📎 Enter File URL (Optional)", placeholder="https://example.com/document.pdf")

if uploaded_file or file_url:
    saved_path = None
    if uploaded_file:
        saved_path = save_uploaded_file(uploaded_file)
    elif file_url:
        saved_path = download_file_from_url(file_url)

    if saved_path:
        file_extension = saved_path.split(".")[-1].lower()

        if file_extension == "pdf":
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(" ".join([doc.page_content for doc in raw_docs]))
        else:
            raw_text = load_text_file(saved_path)
            processed_chunks = chunk_documents(raw_text)

        index_documents(processed_chunks)
        
        st.success(f"✅ Document indexed in ChromaDB using {selected_model}! Ask your questions below.")
        
        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Searching ChromaDB..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)

st.markdown("---")
st.caption("⚡ Built with ChromaDB, LangChain & Open-Source LLMs | gprMax Chatbot 🚀")

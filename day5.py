import os
import streamlit as st
import tempfile
import shutil
import pathlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ---------------------- Configuration ----------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyCf7rxCiOtbLI5SnmjDRV9PS9WxWWk8uTI"  # Replace with your actual key
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="RAG + Memory with Gemini", layout="wide")
st.title("📄 RAG Chat with Memory using Gemini 2.0 Flash")
st.markdown("Upload PDFs and chat using Retrieval-Augmented Generation with memory.")

# ---------------------- Session State ----------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "retriever" not in st.session_state:
    st.session_state.retriever = None

memory = st.session_state.memory

# ---------------------- File Upload ----------------------
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

@st.cache_resource
def process_files(files):
    all_docs = []
    tmp_dir = tempfile.mkdtemp()
    try:
        for file in files:
            temp_path = pathlib.Path(tmp_dir) / file.name
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(str(temp_path))
            all_docs.extend(loader.load())

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever()
    finally:
        shutil.rmtree(tmp_dir)

if uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.retriever = process_files(uploaded_files)
    st.success("Documents processed and indexed!")

retriever = st.session_state.retriever

# ---------------------- Prompt Template ----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer questions."),
    ("human", "Context:\n{context}\n\nChat History:\n{chat_history}\n\nQuestion: {question}")
])

# ---------------------- LLM ----------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ---------------------- RAG Chain ----------------------
def join_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(_):
    return "\n".join(
        f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in memory.chat_memory.messages
    )

chain = (
    {
        "question": RunnablePassthrough(),
        "context": (retriever if retriever else RunnableLambda(lambda _: [])) | RunnableLambda(join_docs),
        "chat_history": RunnableLambda(format_history)
    }
    | prompt
    | llm
)

# ---------------------- Chat Interface ----------------------
user_question = st.text_input("Ask a question:")

if user_question and not retriever:
    st.warning("Please upload and index at least one PDF.")
elif user_question:
    with st.spinner("Generating answer..."):
        response = chain.invoke(user_question)
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(response.content)

    st.markdown("### 🤖 Answer")
    st.write(response.content)

    st.markdown("### 🧠 Chat History")
    for msg in memory.chat_memory.messages:
        role = "🧑 You" if msg.type == "human" else "🤖 AI"
        st.markdown(f"**{role}:** {msg.content}")

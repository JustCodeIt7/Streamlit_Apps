import streamlit as st
import subprocess
import os
import tempfile
import zipfile
from typing import List, Dict

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Define allowed file extensions
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".sql", ".c", ".cpp",
    ".cs", ".go", ".rb", ".php", ".html", ".css", ".md", ".json", ".yml",
    ".yaml", ".xml", ".sh", ".bash", ".txt"
}

def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_codebase(directory: str) -> List[Dict[str, str]]:
    """Load all code files from a directory and extract their content."""
    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext not in ALLOWED_EXTENSIONS or file.startswith('.'):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if content.strip():
                        rel_path = os.path.relpath(file_path, directory)
                        documents.append({
                            "source": rel_path,
                            "content": content
                        })
            except Exception as e:
                st.warning(f"Could not read {file_path}: {e}")

    return documents

def clone_github_repo(repo_url: str, target_dir: str) -> bool:
    """Clone a GitHub repository into the target directory."""
    clone_command = ["git", "clone", "--depth=1", repo_url, target_dir]
    try:
        subprocess.run(clone_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        st.error(f"Error cloning the repository: {e}")
        return False

def process_codebase(docs: List[Dict[str, str]], use_openai: bool = True):
    """Process codebase and create QA chain."""
    # Prepare documents for embedding
    texts = [doc["content"] for doc in docs]
    metadatas = [{"source": doc["source"]} for doc in docs]

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    chunk_metadatas = []
    for text, metadata in zip(texts, metadatas):
        splits = text_splitter.split_text(text)
        chunks.extend(splits)
        chunk_metadatas.extend([metadata] * len(splits))

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings() if use_openai else OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=chunk_metadatas)

    # Create QA chain
    llm = OpenAI(temperature=0) if use_openai else ChatOllama(model="llama3", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )

    return qa_chain

def main():
    st.title("Codebase Documentation Chatbot")
    st.markdown("""
    Ask questions about your codebase to understand its structure, functionality, and documentation.
    Examples:
    - "Where is the function that handles authentication?"
    - "What does the user management module do?"
    - "Explain how the database connection is established."
    """)

    # Sidebar for configuration
    st.sidebar.title("Settings")
    input_method = st.sidebar.radio("Choose input method:", ["Upload ZIP", "GitHub URL"])
    use_openai = st.sidebar.checkbox("Use OpenAI API", value=True)

    if use_openai:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Handle codebase input
    col1, col2 = st.columns(2)

    with col1:
        if input_method == "Upload ZIP":
            uploaded_file = st.file_uploader("Upload codebase ZIP", type=["zip"])
            process_button = st.button("Process Uploaded Codebase")

            if uploaded_file and process_button:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    with st.spinner("Processing uploaded codebase..."):
                        # Save and extract zip file
                        zip_path = os.path.join(tmpdirname, "repo.zip")
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        extract_zip(zip_path, tmpdirname)

                        # Process codebase
                        docs = load_codebase(tmpdirname)
                        if not docs:
                            st.error("No valid code files found in the ZIP archive.")
                            return

                        st.session_state.qa_chain = process_codebase(docs, use_openai)
                        st.success(f"Processed {len(docs)} files from the codebase!")

        else:  # GitHub URL
            repo_url = st.text_input("GitHub Repository URL:", "https://github.com/username/repo")
            process_button = st.button("Clone & Process Repository")

            if repo_url and process_button:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    with st.spinner("Cloning and processing repository..."):
                        if clone_github_repo(repo_url, tmpdirname):
                            # Process codebase
                            docs = load_codebase(tmpdirname)
                            if not docs:
                                st.error("No valid code files found in the repository.")
                                return

                            st.session_state.qa_chain = process_codebase(docs, use_openai)
                            st.success(f"Processed {len(docs)} files from the repository!")

    with col2:
        # Question answering interface
        if st.session_state.qa_chain:
            st.subheader("Ask about the codebase")
            user_question = st.text_input("Your question:", key="question_input")
            if st.button("Get Answer"):
                if user_question:
                    with st.spinner("Analyzing codebase..."):
                        try:
                            answer = st.session_state.qa_chain.run(user_question)
                            st.session_state.chat_history.append({"question": user_question, "answer": answer})
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                else:
                    st.warning("Please enter a question.")
        else:
            st.info("Upload or clone a repository to start asking questions.")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {exchange['question'][:50]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Question:** {exchange['question']}")
                st.markdown(f"**Answer:** {exchange['answer']}")

if __name__ == "__main__":
    main()
import os
import tempfile
import streamlit as st
import git
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Codebase Documentation Chatbot", page_icon="💻", layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False


# Function to extract code documentation and comments
def extract_code_info(file_path):
    """Extract code, documentation, and comments from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


# Function to process a repository
def process_repository(repo_path):
    """Process a repository to extract documentation and code information."""
    # Get all code files
    code_extensions = [
        ".py",
        ".js",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".ts",
        ".html",
        ".css",
    ]
    all_files = []

    for ext in code_extensions:
        all_files.extend(glob.glob(f"{repo_path}/**/*{ext}", recursive=True))

    # Extract content from each file
    documents = []
    for file_path in all_files:
        relative_path = os.path.relpath(file_path, repo_path)
        content = extract_code_info(file_path)
        if content:
            documents.append(f"File: {relative_path}\n\n{content}")

    return documents


# Function to create a vector store from documents
def create_vector_store(documents):
    """Create a vector store from the documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.create_documents([doc for doc in documents])

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# Function to create a conversational chain
def create_conversation_chain(vector_store):
    """Create a conversational chain for the chatbot."""
    llm = ChatOpenAI(model_name="gpt-4")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

    return conversation_chain


# Main app layout
st.title("💻 Codebase Documentation Chatbot")
st.markdown(
    """
This app helps you navigate and understand a software repository by analyzing its documentation and inline comments.
Upload a repository or connect to GitHub to get started.
"""
)

# Sidebar for repository selection
with st.sidebar:
    st.header("Repository Selection")

    repo_option = st.radio("Choose repository source:", ("Upload ZIP", "GitHub URL"))

    if repo_option == "Upload ZIP":
        uploaded_file = st.file_uploader("Upload repository ZIP file", type=["zip"])

        if uploaded_file and not st.session_state.repo_processed:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the uploaded file
                zip_path = os.path.join(temp_dir, "repo.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract the ZIP file
                import zipfile

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find the repository root directory
                subdirs = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
                repo_path = (
                    subdirs[0] if subdirs and subdirs[0] != zip_path else temp_dir
                )

                with st.spinner("Processing repository..."):
                    # Process the repository
                    documents = process_repository(repo_path)

                    if documents:
                        # Create vector store and conversation chain
                        vector_store = create_vector_store(documents)
                        st.session_state.conversation = create_conversation_chain(
                            vector_store
                        )
                        st.session_state.repo_processed = True
                        st.success("Repository processed successfully!")
                    else:
                        st.error("No code files found in the repository.")

    elif repo_option == "GitHub URL":
        github_url = st.text_input(
            "GitHub Repository URL", placeholder="https://github.com/username/repo"
        )
        github_token = st.text_input(
            "GitHub Token (optional, for private repos)", type="password"
        )

        if (
            st.button("Process Repository")
            and github_url
            and not st.session_state.repo_processed
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with st.spinner("Cloning repository..."):
                        # Extract repo information from URL
                        repo_parts = github_url.rstrip("/").split("/")
                        repo_name = f"{repo_parts[-2]}/{repo_parts[-1]}"

                        # Clone the repository
                        if github_token:
                            clone_url = (
                                f"https://{github_token}@github.com/{repo_name}.git"
                            )
                        else:
                            clone_url = f"https://github.com/{repo_name}.git"

                        git.Repo.clone_from(clone_url, temp_dir)

                        # Process the repository
                        documents = process_repository(temp_dir)

                        if documents:
                            # Create vector store and conversation chain
                            vector_store = create_vector_store(documents)
                            st.session_state.conversation = create_conversation_chain(
                                vector_store
                            )
                            st.session_state.repo_processed = True
                            st.success("Repository processed successfully!")
                        else:
                            st.error("No code files found in the repository.")

                except Exception as e:
                    st.error(f"Error processing repository: {str(e)}")

    if st.session_state.repo_processed:
        if st.button("Reset"):
            st.session_state.repo_processed = False
            st.session_state.conversation = None
            st.session_state.messages = []
            st.rerun()

# Chat interface
if st.session_state.repo_processed:
    st.header("Ask about the codebase")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about the codebase...")

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get response from conversation chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke(
                    {"question": user_question}
                )
                response_text = response["answer"]
                st.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
else:
    st.info("Please select a repository to get started.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and OpenAI")

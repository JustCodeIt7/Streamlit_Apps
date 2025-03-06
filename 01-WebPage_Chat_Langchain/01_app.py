from numpy import imag
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # if you wannt to use openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ============= Configuration Component =============
def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(page_title="Webpage Chat", page_icon="🕸️", layout="wide")
    st.title("Webpage Chat with LangChain & Ollama")

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def create_sidebar_config():
    """Create and return the configuration from the sidebar."""
    with st.sidebar:
        st.header("Configuration")
        config = {
            "ollama_model": st.selectbox(
                "Select Ollama Model",
                ["deepseek-r1:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "deepseek-r1:8b"],
                index=0
            ),
            "embedding_model": st.selectbox(
                "Select Embedding Model",
                ["nomic-embed-text", "snowflake-arctic-embed:latest"],
                index=0
            ),
            "text_threshold": st.slider(
                "Text size threshold (characters) for vector DB vs. full context",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            ),
            "chunk_size": st.slider(
                "Chunk size for text splitting",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100
            ),
            "chunk_overlap": st.slider(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=50,
                step=10
            )
        }
        return config

# ============= Web Processing Component =============
def extract_text_from_webpage(url):
    """
    Extract and clean text from a webpage.

    Args:
        url (str): The URL of the webpage to extract text from.

    Returns:
        str or None: The extracted text, or None if an error occurred.
    """
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content
    except Exception as e:
        st.error(f"Error loading webpage: {e}")
        return None

def create_full_context_processor(llm, text):
    """
    Create a processor for the full context approach.

    Args:
        llm (ChatOllama): The language model.
        text (str): The full text from the webpage.

    Returns:
        function: A function that processes user queries using the full context.
    """
    def process_query(user_input):
        prompt = f"""
        You are an AI assistant that helps users understand webpage content.

        Webpage content:
        {text}

        User question: {user_input}

        Please provide a helpful, accurate, and concise answer based on the webpage content.
        """
        response = llm.invoke(prompt)
        return response.content

    return process_query

def create_vector_db_processor(qa_chain):
    """
    Create a processor for the vector database approach.

    Args:
        qa_chain (ConversationalRetrievalChain): The conversational retrieval chain.

    Returns:
        function: A function that processes user queries using the vector database.
    """
    def process_query(user_input):
        response = qa_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        response_text = response["answer"]
        st.session_state.chat_history.append((user_input, response_text))
        return response_text

    return process_query

def setup_vector_approach(text, config):
    """
    Set up the vector database approach for large texts.

    Args:
        text (str): The text to process.
        config (dict): Configuration parameters.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain.
    """
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the conversational chain
    llm = ChatOllama(model=config["ollama_model"])
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return qa_chain

def process_webpage(url, config):
    """
    Process a webpage by extracting its text and setting up a query processor.

    Args:
        url (str): The URL of the webpage to process.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing:
            - function: The query processor function.
            - int: The length of the extracted text.
    """
    with st.spinner("Processing webpage..."):
        text = extract_text_from_webpage(url)

        if text is None:
            return None, 0

        text_length = len(text)
        st.info(f"Extracted {text_length} characters from the webpage")

        # Initialize the LLM
        llm = ChatOllama(model=config["ollama_model"])

        # If text is smaller than threshold, use full context approach
        if text_length < config["text_threshold"]:
            st.success("Using full context approach (text is relatively small)")
            processor = create_full_context_processor(llm, text)
            st.session_state.full_text = text
            st.session_state.vector_approach = False
        else:
            # Otherwise, use vector embeddings approach
            st.success("Using vector embeddings approach (text is relatively large)")
            qa_chain = setup_vector_approach(text, config)
            processor = create_vector_db_processor(qa_chain)
            st.session_state.full_text = None
            st.session_state.vector_approach = True

        return processor, text_length

# ============= Chat UI Component =============
def display_chat_messages():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_input(query_processor):
    """
    Handle user input and generate responses.

    Args:
        query_processor (function): The function to process user queries.
    """
    user_input = st.chat_input("Ask a question about the webpage:")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = query_processor(user_input)
                st.write(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# ============= Main Application =============
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Setup page configuration
    setup_page_config()

    # Get configuration from sidebar
    config = create_sidebar_config()


        # Main app interface
    url_input = st.text_input("Enter a webpage URL:", "https://python.langchain.com/docs/get_started/introduction")

    if st.button("Process Webpage"):
        if url_input:
            # Store the URL in session state
            st.session_state.url = url_input

            # Process the webpage
            query_processor, text_length = process_webpage(url_input, config)

            if query_processor is not None:
                st.session_state.query_processor = query_processor
                st.session_state.text_length = text_length
                st.session_state.chat_history = []
                st.session_state.messages = [{"role": "assistant", "content": "Webpage processed! You can now ask questions about it."}]
        else:
            st.warning("Please enter a valid URL")


    # Display chat messages
    display_chat_messages()

    # Handle user input if a webpage has been processed
    if "query_processor" in st.session_state:
        handle_user_input(st.session_state.query_processor)

if __name__ == "__main__":
    main()

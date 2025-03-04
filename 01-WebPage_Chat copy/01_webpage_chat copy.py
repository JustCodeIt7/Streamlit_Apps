import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="Chat with Webpages", page_icon="🌐", layout="wide")
st.title("Chat with Webpages using LangChain and Ollama")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    ollama_model = st.selectbox(
        "Select Ollama Model",
        ["deepseek-r1:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "deepseek-r1:8b"],
        index=0
    )
    
    embedding_model = st.selectbox(
        "Select Embedding Model",
        ["nomic-embed-text", "snowflake-arctic-embed:latest"],
        index=0
    )
    
    text_threshold = st.slider(
        "Text size threshold (characters) for vector DB vs. full context",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    chunk_size = st.slider(
        "Chunk size for text splitting",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=50,
        step=10
    )

# Function to extract and clean text from a webpage
def extract_text_from_webpage(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content
    except Exception as e:
        st.error(f"Error loading webpage: {e}")
        return None

# Function to process the webpage content
def process_webpage(url):
    with st.spinner("Processing webpage..."):
        text = extract_text_from_webpage(url)
        if text is None:
            return None, None, 0
        
        text_length = len(text)
        st.info(f"Extracted {text_length} characters from the webpage")
        
        # Initialize the LLM
        llm = ChatOllama(model=ollama_model)
        
        # If text is smaller than threshold, use full context approach
        if text_length < text_threshold:
            st.success("Using full context approach (text is relatively small)")
            return llm, text, text_length
        
        # Otherwise, use vector embeddings approach
        st.success("Using vector embeddings approach (text is relatively large)")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        
        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        
        return qa_chain, None, text_length

# Main app interface
url_input = st.text_input("Enter a webpage URL:", "https://python.langchain.com/docs/get_started/introduction")

if st.button("Process Webpage"):
    if url_input:
        # Store the URL in session state
        st.session_state.url = url_input
        
        # Process the webpage
        chain_or_llm, full_text, text_length = process_webpage(url_input)
        
        if chain_or_llm is not None:
            st.session_state.chain_or_llm = chain_or_llm
            st.session_state.full_text = full_text
            st.session_state.text_length = text_length
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "Webpage processed! You can now ask questions about it."}]
    else:
        st.warning("Please enter a valid URL")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if "chain_or_llm" in st.session_state:
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
                if st.session_state.full_text is not None:
                    # Full context approach
                    prompt = f"""
                    You are an AI assistant that helps users understand webpage content.
                    
                    Webpage content:
                    {st.session_state.full_text}
                    
                    User question: {user_input}
                    
                    Please provide a helpful, accurate, and concise answer based on the webpage content.
                    """
                    response = st.session_state.chain_or_llm.invoke(prompt)
                    response_text = response.content
                else:
                    # Vector DB approach
                    response = st.session_state.chain_or_llm.invoke({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    response_text = response["answer"]
                    st.session_state.chat_history.append((user_input, response_text))
                
                st.write(response_text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

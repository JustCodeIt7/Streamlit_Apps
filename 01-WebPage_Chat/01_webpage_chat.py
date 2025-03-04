# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Optional, List, Dict, Any
import uuid

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# LangGraph imports
import langgraph.graph as lg
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


# ============================================================
# DATA STRUCTURES
# ============================================================


class WebpageChatState(TypedDict):
    """State structure for the LangGraph workflow."""

    url: str
    webpage_content: Optional[str]
    approach: Optional[str]
    error: Optional[str]


class ChatState(TypedDict):
    """State structure for the chat workflow."""

    messages: List[BaseMessage]
    context: Optional[str]
    question: Optional[str]
    answer: Optional[str]


# ============================================================
# WEBPAGE CONTENT EXTRACTION
# ============================================================


def extract_webpage_content(url):
    """
    Extract and clean text content from a webpage.

    Args:
        url: The URL of the webpage to extract content from

    Returns:
        tuple: (extracted_text, error_message)
    """
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text content
        text = soup.get_text()

        # Clean the extracted text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text, None
    except Exception as e:
        return None, f"Error fetching webpage: {str(e)}"

# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================


def fetch_webpage(state: WebpageChatState) -> WebpageChatState:
    """
    LangGraph node: Fetch content from a webpage.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    url = state["url"]
    content, error = extract_webpage_content(url)

    if error:
        return {**state, "error": error}

    return {**state, "webpage_content": content, "error": None}




def determine_approach(state: WebpageChatState) -> WebpageChatState:
    """
    LangGraph node: Determine whether to use full context or embeddings.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with chosen approach
    """
    if state.get("error"):
        return state

    content = state["webpage_content"]
    threshold = st.session_state.get("text_threshold", 4000)

    # Choose approach based on content length
    approach = "full_context" if len(content) < threshold else "embeddings"

    return {**state, "approach": approach}


def build_webpage_processor():
    """
    Build a LangGraph workflow for processing webpages.

    Returns:
        Compiled LangGraph workflow
    """
    # Create a new state graph
    workflow = lg.StateGraph(WebpageChatState)

    # Add nodes to the graph
    workflow.add_node("fetch_webpage", fetch_webpage)
    workflow.add_node("determine_approach", determine_approach)

    # Add edges between nodes
    workflow.add_edge("fetch_webpage", "determine_approach")

    # Set the entry point
    workflow.set_entry_point("fetch_webpage")

    # Compile and return the graph
    return workflow.compile()

# ============================================================
# CHAT ENGINES
# ============================================================

def format_chat_history(chat_history):
    """
    Format chat history for context.
    
    Args:
        chat_history: List of message dictionaries
        
    Returns:
        Formatted chat history string
    """
    chat_context = ""
    if chat_history:
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_context += f"{role}: {msg['content']}\n"
    return chat_context


def generate_full_context_response(query, chat_history, content, llm):
    """
    Generate a response using the full webpage content.
    
    Args:
        query: User's question
        chat_history: Previous conversation
        content: Webpage content
        llm: Language model
        
    Returns:
        Generated response
    """
    chat_context = format_chat_history(chat_history)
    
    prompt = f"""
    You are an AI assistant that helps users understand and extract information from webpages.

    Webpage content:
    {content}
    
    Previous conversation:
    {chat_context}

    Answer the user's question based on this webpage content. If the information is not in the content, say so.

    User's question: {query}
    """
    return llm.invoke(prompt)
    



def generate_embeddings_response(query, chat_history, retriever, llm):
    """
    Generate a response using retrieval with embeddings.
    
    Args:
        query: User's question
        chat_history: Previous conversation
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Generated response
    """
    chat_context = format_chat_history(chat_history)
    
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)
    retrieved_content = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are an AI assistant that helps users understand and extract information from webpages.

    Relevant webpage content:
    {retrieved_content}
    
    Previous conversation:
    {chat_context}

    Answer the user's question based on this webpage content. If the information is not in the content, say so.

    User's question: {query}
    """
    return llm.invoke(prompt)
    



def setup_full_context_chat(content, model_name):
    """
    Set up a chat engine that uses the full webpage content.

    Args:
        content: The webpage content
        model_name: The Ollama model to use

    Returns:
        Function that generates responses to queries
    """
    # Initialize the LLM
    llm = OllamaLLM(model=model_name)
    
    # Return a function that closes over the content and llm
    def response_generator(query, chat_history=None):
        if chat_history is None:
            chat_history = []
        return generate_full_context_response(query, chat_history, content, llm)
    
    return response_generator




def setup_embeddings_chat(content, model_name):
    """
    Set up a chat engine that uses embeddings and retrieval.

    Args:
        content: The webpage content
        model_name: The Ollama model to use

    Returns:
        Function that generates responses to queries
    """
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(content)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize the LLM
    llm = OllamaLLM(model=model_name)
    
    # Return a function that closes over the retriever and llm
    def response_generator(query, chat_history=None):
        if chat_history is None:
            chat_history = []
        return generate_embeddings_response(query, chat_history, retriever, llm)
    
    return response_generator



# ============================================================
# STREAMLIT UI COMPONENTS
# ============================================================
def initialize_session_state():
    """Initialize session state variables."""
    if "current_model" not in st.session_state:
        st.session_state.current_model = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False

    if "webpage_content" not in st.session_state:
        st.session_state.webpage_content = None


def render_sidebar():
    """Render the sidebar with settings and controls."""
    st.sidebar.header("Settings")

    # Model selection dropdown
    model_options = [
        "deepseek-r1:1.5b",
        "llama3.2:1b",
        "deepseek-r1:14b",
        "qwen2.5:0.5b",
    ]
    model_name = st.sidebar.selectbox(
        "Select Ollama model:",
        model_options,
        index=0,
    )

    # Detect model change
    model_changed = st.session_state.current_model != model_name
    st.session_state.current_model = model_name
    st.session_state.model_name = model_name

    # Text threshold slider
    text_threshold = st.sidebar.slider(
        "Text length threshold (characters)",
        min_value=1000,
        max_value=10000,
        value=4000,
        step=500,
        help="If webpage text exceeds this length, embeddings will be used instead of full context",
    )
    st.session_state.text_threshold = text_threshold

    # Add option to clear chat history
    clear_history = st.sidebar.button("Clear Chat History")

    return {
        "model_name": model_name,
        "model_changed": model_changed,
        "text_threshold": text_threshold,
        "clear_history": clear_history,
    }


def render_url_input():
    """Render the URL input field and process button."""
    url = st.text_input("Enter webpage URL:", "https://docs.streamlit.io/")
    process_button = st.button("Process Webpage")

    return {"url": url, "process_button": process_button}


def process_webpage(url, model_name, text_threshold):
    """Process a webpage and set up the chat engine."""
    with st.spinner("Processing webpage..."):
        # Set up and run the LangGraph workflow
        processor = build_webpage_processor()

        # Process the webpage
        result = processor.invoke(
            {"url": url, "webpage_content": None, "approach": None, "error": None}
        )

        # Handle errors
        if result.get("error"):
            st.error(result["error"])
            return False

        # Reset chat history
        st.session_state.messages = []

        # Store the webpage content and approach
        content = result["webpage_content"]
        approach = "full_context" if len(content) < text_threshold else "embeddings"

        # Store content for potential reprocessing
        st.session_state.webpage_content = content
        st.session_state.webpage_url = url

        # Set up appropriate chat engine based on approach
        if approach == "full_context":
            st.session_state.chat_engine = setup_full_context_chat(content, model_name)
        else:  # embeddings approach
            st.session_state.chat_engine = setup_embeddings_chat(content, model_name)

        # Update session state
        st.session_state.chat_initialized = True
        st.session_state.approach = approach
        st.success(f"Webpage processed successfully! Using {approach} approach.")

        return True



def reinitialize_chat_engine(model_name, text_threshold):
    """Reinitialize the chat engine with a new model."""
    with st.spinner(f"Reinitializing with model {model_name}..."):
        content = st.session_state.webpage_content
        approach = "full_context" if len(content) < text_threshold else "embeddings"

        # Set up appropriate chat engine based on approach
        if approach == "full_context":
            st.session_state.chat_engine = setup_full_context_chat(content, model_name)
        else:  # embeddings approach
            st.session_state.chat_engine = setup_embeddings_chat(content, model_name)

        # Update session state
        st.session_state.approach = approach
        st.success(f"Chat engine reinitialized with model {model_name}!")
        

def render_chat_interface():
    """Render the chat interface for interacting with the webpage."""
    st.info(
        f"Current approach: {st.session_state.approach} | Model: {st.session_state.current_model}"
    )

    if hasattr(st.session_state, "webpage_url"):
        st.info(f"Current webpage: {st.session_state.webpage_url}")

    # Display chat message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input for user queries
    user_input = st.chat_input("Ask about the webpage:")

    if user_input:
        process_user_input(user_input)


def process_user_input(user_input):
    """Process user input and generate a response."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response based on the chosen approach
            response = st.session_state.chat_engine(
                user_input, st.session_state.messages[:-1]
            )

            # Display the response
            st.write(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})




def render_welcome_message():
    """Render a welcome message when no webpage is loaded."""
    st.markdown(
        """
    ### 👋 Welcome to Webpage Chat!
    
    This app allows you to chat with any webpage using Ollama models.
    
    **To get started:**
    1. Enter a URL in the text field above
    2. Click "Process Webpage"
    3. Ask questions about the webpage content
    
    The app will automatically choose between using the full webpage content or 
    creating embeddings based on the content length.
    """
    )


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main Streamlit application."""
    st.title("Webpage Chat with LangGraph and Ollama")

    # Initialize session state
    initialize_session_state()
    

    # Render sidebar and get settings
    sidebar_config = render_sidebar()
    model_name = sidebar_config["model_name"]
    model_changed = sidebar_config["model_changed"]
    text_threshold = sidebar_config["text_threshold"]
    
    # Handle clear history button
    if sidebar_config["clear_history"]:
        st.session_state.messages = []
        st.rerun()
    
    # Render URL input and get values
    input_config = render_url_input()
    url = input_config["url"]
    process_button = input_config["process_button"]
    
    
    # Determine if we need to reprocess due to model change
    reprocess = model_changed and st.session_state.chat_initialized
    if reprocess:
        st.warning(f"Model changed to {model_name}. Reinitializing chat engine...")
    
    # Process webpage or reinitialize chat engine
    if process_button:
        process_webpage(url, model_name, text_threshold)
    elif reprocess and st.session_state.webpage_content:
        reinitialize_chat_engine(model_name, text_threshold)

    # Render chat interface or welcome message
    if st.session_state.chat_initialized:
        render_chat_interface()
    else:
        render_welcome_message()

    
# ============================================================
# APP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

# LangGraph imports
import langgraph.graph as lg


# ============================================================
# DATA STRUCTURES
# ============================================================
class WebpageChatState(TypedDict):
    """State structure for the LangGraph workflow."""

    url: str
    webpage_content: Optional[str]
    approach: Optional[str]
    error: Optional[str]


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
        response = requests.get(url)
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

    def get_response(query):
        """Generate a response to the user's query."""
        prompt = f"""
        You are an AI assistant that helps users understand and extract information from webpages.

        Webpage content:
        {content}

        Answer the user's question based on this webpage content. If the information is not in the content, say so.

        User's question: {query}
        """
        return llm.invoke(prompt)

    return get_response


def setup_embeddings_chat(content, model_name):
    """
    Set up a chat engine that uses embeddings and retrieval.

    Args:
        content: The webpage content
        model_name: The Ollama model to use

    Returns:
        ConversationalRetrievalChain for answering queries
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

    # Create memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create and return the conversational chain
    llm = OllamaLLM(model=model_name)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    return qa_chain


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    """Main Streamlit application."""
    st.title("Webpage Chat with LangGraph and Ollama")

    # ---- SIDEBAR CONFIGURATION ----
    st.sidebar.header("Settings")

    # Model selection dropdown
    model_name = st.sidebar.selectbox(
        "Select Ollama model:",
        ["deepseek-r1:1.5b", "llama3.2:1b", "deepseek-r1:14b", "qwen2.5:0.5b"],
        index=0,
    )
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

    # ---- MAIN UI ----
    # URL input field
    url = st.text_input("Enter webpage URL:", "https://example.com")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False

    # ---- WEBPAGE PROCESSING ----
    if st.button("Process Webpage"):
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
            else:
                # Reset chat history
                st.session_state.messages = []

                # Store the webpage content and approach
                content = result["webpage_content"]
                approach = result["approach"]

                # Set up appropriate chat engine based on approach
                if approach == "full_context":
                    st.session_state.chat_engine = setup_full_context_chat(
                        content, model_name
                    )
                else:  # embeddings approach
                    st.session_state.chat_engine = setup_embeddings_chat(
                        content, model_name
                    )

                # Update session state
                st.session_state.chat_initialized = True
                st.session_state.approach = approach
                st.success(
                    f"Webpage processed successfully! Using {approach} approach."
                )

    # ---- CHAT INTERFACE ----
    if st.session_state.chat_initialized:
        st.info(f"Current approach: {st.session_state.approach}")

        # Display chat message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input for user queries
        user_input = st.chat_input("Ask about the webpage:")

        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message
            with st.chat_message("user"):
                st.write(user_input)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response based on the chosen approach
                    if st.session_state.approach == "full_context":
                        response = st.session_state.chat_engine(user_input)
                    else:  # embeddings approach
                        result = st.session_state.chat_engine.invoke(
                            {"question": user_input}
                        )
                        response = result["answer"]

                    # Display the response
                    st.write(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )


# ============================================================
# APP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import langgraph.graph as lg
from typing import TypedDict, Optional


# Define the state structure for LangGraph
class WebpageChatState(TypedDict):
    url: str
    webpage_content: Optional[str]
    approach: Optional[str]
    error: Optional[str]


# Function to extract text from a webpage
def extract_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text()

        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text, None
    except Exception as e:
        return None, f"Error fetching webpage: {str(e)}"


# Define LangGraph nodes as normal functions (without decorators)
def fetch_webpage(state: WebpageChatState) -> WebpageChatState:
    """Fetch content from a webpage."""
    url = state["url"]
    content, error = extract_webpage_content(url)

    if error:
        return {**state, "error": error}

    return {**state, "webpage_content": content, "error": None}


def determine_approach(state: WebpageChatState) -> WebpageChatState:
    """Determine whether to use full context or embeddings."""
    if state.get("error"):
        return state

    content = state["webpage_content"]
    threshold = st.session_state.get("text_threshold", 4000)

    approach = "full_context" if len(content) < threshold else "embeddings"

    return {**state, "approach": approach}


# Build the LangGraph workflow
def build_webpage_processor():
    """Build a graph for processing webpages."""
    workflow = lg.StateGraph(WebpageChatState)

    # Add nodes to the graph explicitly (instead of using decorators)
    workflow.add_node("fetch_webpage", fetch_webpage)
    workflow.add_node("determine_approach", determine_approach)

    # Add edges
    workflow.add_edge("fetch_webpage", "determine_approach")

    # Set entry point
    workflow.set_entry_point("fetch_webpage")

    # Compile the graph
    return workflow.compile()


# Function to set up full context chat
def setup_full_context_chat(content, model_name):
    llm = Ollama(model=model_name)

    def get_response(query):
        prompt = f"""
        You are an AI assistant that helps users understand and extract information from webpages.

        Webpage content:
        {content}

        Answer the user's question based on this webpage content. If the information is not in the content, say so.

        User's question: {query}
        """
        return llm.invoke(prompt)

    return get_response


# Function to set up embeddings-based chat
def setup_embeddings_chat(content, model_name):
    # Split text into chunks
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

    # Create conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        "Select Ollama model:",
        ["deepseek-r1:1.5b", "llama3.2:1b", "deepseek-r1:14b", "qwen2.5:0.5b"],
        index=0,
    )

    return qa_chain


# Streamlit app
def main():
    st.title("Webpage Chat with LangGraph and Ollama")

    # Sidebar for configuration
    st.sidebar.header("Settings")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Ollama model:",
        ["deepseek-r1:1.5b", "llama3.2:1b", "deepseek-r1:14b", "qwen2.5:0.5b"],
        index=0,
    )
    st.session_state.model_name = model_name

    # Text threshold setting
    text_threshold = st.sidebar.slider(
        "Text length threshold (characters)",
        min_value=1000,
        max_value=10000,
        value=4000,
        step=500,
        help="If webpage text exceeds this length, embeddings will be used instead of full context",
    )
    st.session_state.text_threshold = text_threshold

    # URL input
    url = st.text_input("Enter webpage URL:", "https://example.com")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False

    # Process button
    if st.button("Process Webpage"):
        with st.spinner("Processing webpage..."):
            # Set up the LangGraph workflow
            processor = build_webpage_processor()

            # Process the webpage using LangGraph
            result = processor.invoke(
                {"url": url, "webpage_content": None, "approach": None, "error": None}
            )

            if result.get("error"):
                st.error(result["error"])
            else:
                # Reset chat
                st.session_state.messages = []

                # Store the webpage content and approach
                content = result["webpage_content"]
                approach = result["approach"]

                # Set up chat based on approach
                if approach == "full_context":
                    st.session_state.chat_engine = setup_full_context_chat(
                        content, model_name
                    )
                else:
                    st.session_state.chat_engine = setup_embeddings_chat(
                        content, model_name
                    )

                st.session_state.chat_initialized = True
                st.session_state.approach = approach
                st.success(
                    f"Webpage processed successfully! Using {approach} approach."
                )

    # Display chat interface if initialized
    if st.session_state.chat_initialized:
        st.info(f"Current approach: {st.session_state.approach}")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input for user query
        user_input = st.chat_input("Ask about the webpage:")

        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response based on approach
                    if st.session_state.approach == "full_context":
                        response = st.session_state.chat_engine(user_input)
                    else:  # embeddings approach
                        result = st.session_state.chat_engine.invoke(
                            {"question": user_input}
                        )
                        response = result["answer"]

                    # Display response
                    st.write(response)

                    # Add assistant response to chat
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )


if __name__ == "__main__":
    main()

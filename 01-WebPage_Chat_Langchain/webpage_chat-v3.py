import streamlit as st
import ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph
from typing import Dict, List, TypedDict
import re

# Set page title
st.set_page_config(page_title="Chat with Webpage", page_icon="🌐")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )
if "webpage_content" not in st.session_state:
    st.session_state.webpage_content = None
if "use_embeddings" not in st.session_state:
    st.session_state.use_embeddings = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Title and description
st.title("Chat with Webpage 🌐")
st.markdown(
    "Enter a URL and chat with its content using LangChain, LangGraph, and Ollama."
)

# Input for URL
url_input = st.text_input("Enter webpage URL:", placeholder="https://example.com")

# Model selection
model_name = st.selectbox(
    "Select Ollama model:",
    ["deepseek-r1:1.5b", "llama3.2:1b", "deepseek-r1:14b", "qwen2.5:0.5b"],
    index=0,
)


# Function to validate and format URL
def format_url(url):
    """Add https:// prefix if the URL doesn't have a schema."""
    if url and not re.match(r"^https?://", url):
        return f"https://{url}"
    return url


# Function to load webpage content
def load_webpage(url):
    # Format URL to ensure it has a schema
    formatted_url = format_url(url)

    with st.spinner(f"Loading webpage content from {formatted_url}..."):
        try:
            loader = WebBaseLoader(formatted_url)
            documents = loader.load()

            # Get the total text content
            full_text = " ".join([doc.page_content for doc in documents])
            text_length = len(full_text)

            # Decide whether to use embeddings based on text length
            use_embeddings = text_length > 10000  # Use embeddings for longer texts

            if use_embeddings:
                st.info(
                    f"Using embeddings for this webpage (text length: {text_length} characters)"
                )
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)

                # Create embeddings and vectorstore
                embeddings = OllamaEmbeddings(model=model_name)
                vectorstore = Chroma.from_documents(
                    documents=splits, embedding=embeddings
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.webpage_content = full_text
                st.session_state.use_embeddings = True
            else:
                st.info(
                    f"Using full text for this webpage (text length: {text_length} characters)"
                )
                st.session_state.webpage_content = full_text
                st.session_state.use_embeddings = False
                st.session_state.vectorstore = None

        except Exception as e:
            st.error(f"Error loading webpage: {str(e)}")
            st.error(
                "Please check the URL and try again. Make sure to include 'https://' if needed."
            )


# Process URL button
if st.button("Process Webpage") and url_input:
    load_webpage(url_input)


# Define state type for LangGraph
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: List


# Define nodes for LangGraph
def retrieve(state: GraphState) -> GraphState:
    """Retrieve relevant context based on the question."""
    question = state["question"]

    if st.session_state.use_embeddings and st.session_state.vectorstore:
        # Use embeddings for retrieval
        docs = st.session_state.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        # Use full text as context
        context = st.session_state.webpage_content

    return {"context": context, **state}


def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer based on the question and context."""
    context = state["context"]
    question = state["question"]
    chat_history = state["chat_history"]

    # Format chat history
    formatted_chat_history = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_chat_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_chat_history += f"AI: {message.content}\n"

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    You are an assistant that answers questions about webpages.
    
    Chat History:
    {chat_history}
    
    Context from the webpage:
    {context}
    
    Question: {question}
    
    Answer the question based only on the provided context. If you don't know the answer, say so.
    """
    )

    # Create LLM
    llm = ChatOllama(model=model_name, temperature=0.1)

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Run chain
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": formatted_chat_history,
        }
    )

    return {"answer": answer, **state}


# Build LangGraph
def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.set_finish_point("generate_answer")

    # Compile graph
    return workflow.compile()


# Chat interface
if st.session_state.webpage_content:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user question
    user_question = st.chat_input("Ask something about the webpage...")

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get chat history for context
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Create and run graph
        graph = build_graph()
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke(
                    {
                        "question": user_question,
                        "chat_history": chat_history,
                        "context": "",
                        "answer": "",
                    }
                )

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(result["answer"])

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please enter a URL and click 'Process Webpage' to start chatting.")

# Add sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
    1. Enter a webpage URL in the text field (e.g., example.com or https://example.com)
    2. Click 'Process Webpage' to load the content
    3. Ask questions about the webpage content
    
    **How it works:**
    - For shorter webpages, the entire content is used as context
    - For longer webpages, the content is split and stored using embeddings for more efficient retrieval
    - LangGraph orchestrates the workflow between retrieval and answer generation
    
    **Requirements:**
    - Ollama running locally (https://ollama.ai)
    - Required models pulled via Ollama
    """
    )

import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Import Ollama wrappers for embeddings and chat models
from langchain_ollama import ChatOllama, OllamaEmbeddings


# --- Functions to Crawl and Process the Website / Page ---


def get_website_text(url: str) -> str:
    """
    Crawl the given URL and extract visible text.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract and clean the text
    text = soup.get_text(separator=" ")
    cleaned_text = " ".join(text.split())
    return cleaned_text


def split_text(text: str):
    """
    Split the text into smaller chunks.
    """
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(chunks):
    """
    Create a FAISS vector store from text chunks using Ollama embeddings.
    """
    # Replace with your Ollama embedding model name.
    embeddings = OllamaEmbeddings(model="all-minilm:33m")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def answer_question(query: str, vectorstore) -> str:
    """
    Retrieve relevant text chunks via similarity search and use an Ollama LLM
    to generate an answer.
    """
    # Retrieve top k relevant chunks
    docs = vectorstore.similarity_search(query, k=4)

    # Instantiate the Ollama chat model.
    # Replace "ollama-chat-model" with your Ollama chat model name.
    llm = ChatOllama(model="llama3.2", temperature=0)

    # Load a QA chain that stuffs the retrieved documents into the prompt.
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    return answer


# --- Streamlit App Layout ---

st.title("Website Chatbot with LangChain and Ollama")

# Sidebar: Select mode and load a page accordingly.
mode = st.sidebar.radio("Select Mode", options=["Crawl Website", "Chat Single Page"])

if mode == "Crawl Website":
    website_url = st.sidebar.text_input(
        "Enter the URL of the website to crawl:", value="https://example.com"
    )
    if st.sidebar.button("Crawl Website"):
        with st.spinner("Crawling website..."):
            website_text = get_website_text(website_url)
            if website_text:
                chunks = split_text(website_text)
                vectorstore = create_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Website crawled and loaded successfully!")
            else:
                st.error("Failed to retrieve website text.")

elif mode == "Chat Single Page":
    website_url = st.sidebar.text_input(
        "Enter the URL of the single page to load:", value="https://example.com"
    )
    if st.sidebar.button("Load Single Page"):
        with st.spinner("Loading single page..."):
            website_text = get_website_text(website_url)
            if website_text:
                chunks = split_text(website_text)
                vectorstore = create_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Single page loaded successfully!")
            else:
                st.error("Failed to retrieve page text.")

# Initialize chat history if not already set.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Chat with the Loaded Content")

if "vectorstore" in st.session_state:
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Your question:")
        submitted = st.form_submit_button("Send")

    if submitted and user_question:
        with st.spinner("Generating answer..."):
            answer = answer_question(user_question, st.session_state.vectorstore)
            # Append the QA pair to the chat history.
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
else:
    st.info("Please load a website or single page from the sidebar to start chatting.")

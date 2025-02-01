import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# --- Functions to Crawl and Process the Website ---


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

    # Extract text and clean it up
    text = soup.get_text(separator=" ")
    cleaned_text = " ".join(text.split())
    return cleaned_text


def split_text(text: str):
    """
    Split the large text into smaller chunks.
    Adjust chunk_size and overlap as needed.
    """
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(chunks):
    """
    Create a FAISS vector store from text chunks using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def answer_question(query: str, vectorstore) -> str:
    """
    Retrieve relevant text chunks via similarity search and use an LLM
    to generate an answer.
    """
    # Retrieve top k relevant chunks
    docs = vectorstore.similarity_search(query, k=4)

    # Load a QA chain – here we use the "stuff" chain type which
    # simply stuffs all the documents into the prompt.
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    return answer


# --- Streamlit App Layout ---

st.title("Website Chatbot with LangChain")

# Sidebar for website URL input
st.sidebar.header("Step 1: Load a Website")
website_url = st.sidebar.text_input(
    "Enter the URL of a single webpage:", value="https://example.com"
)

if st.sidebar.button("Load Website"):
    with st.spinner("Crawling website..."):
        website_text = get_website_text(website_url)
        if website_text:
            chunks = split_text(website_text)
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.success("Website loaded successfully!")
        else:
            st.error("Failed to retrieve website text.")

# Initialize chat history if not already set
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Step 2: Chat with the Website")

if "vectorstore" in st.session_state:
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Your question about the website:")
        submitted = st.form_submit_button("Send")

    if submitted and user_question:
        with st.spinner("Generating answer..."):
            answer = answer_question(user_question, st.session_state.vectorstore)
            # Append to chat history
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
else:
    st.info("Please enter a website URL in the sidebar and click 'Load Website' to begin.")

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ============= Configuration Component =============
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "crawled_urls" not in st.session_state:
        st.session_state.crawled_urls = set()
    if "crawl_in_progress" not in st.session_state:
        st.session_state.crawl_in_progress = False


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(page_title="Chat with Websites", page_icon="🌐", layout="wide")
    st.title("Chat with Websites using LangChain and Ollama")


def create_sidebar_config():
    """Create and return the configuration from the sidebar."""
    with st.sidebar:
        st.header("Configuration")
        config = {
            "ollama_model": st.selectbox(
                "Select Ollama Model",
                ["deepseek-r1:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "deepseek-r1:8b"],
                index=0,
            ),
            "embedding_model": st.selectbox(
                "Select Embedding Model",
                ["nomic-embed-text", "snowflake-arctic-embed:latest"],
                index=0,
            ),
            "text_threshold": st.slider(
                "Text size threshold (characters) for vector DB vs. full context",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
            ),
            "chunk_size": st.slider(
                "Chunk size for text splitting",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
            ),
            "chunk_overlap": st.slider(
                "Chunk overlap", min_value=0, max_value=500, value=50, step=10
            ),
            "crawl_depth": st.slider(
                "Crawl depth",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="How many links deep to crawl from the starting URL",
            ),
            "max_pages": st.slider(
                "Maximum pages to crawl",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Maximum number of pages to crawl",
            ),
            "same_domain_only": st.checkbox(
                "Crawl same domain only",
                value=True,
                help="Only crawl pages from the same domain as the starting URL",
            ),
        }
        return config


# ============= Web Crawling Component =============
def get_domain(url):
    """Extract the domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc


def is_valid_url(url, base_domain, same_domain_only):
    """Check if a URL is valid for crawling."""
    # Basic URL validation
    if not url or not url.startswith(("http://", "https://")):
        return False

    # Check if URL is from the same domain if required
    if same_domain_only:
        url_domain = get_domain(url)
        if url_domain != base_domain:
            return False

    # Skip URLs that are likely to be files or non-HTML content
    file_extensions = [
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".zip",
        ".tar",
        ".gz",
    ]
    if any(url.lower().endswith(ext) for ext in file_extensions):
        return False

    return True


def extract_links_from_page(url, base_domain, same_domain_only):
    """Extract all links from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Handle relative URLs
            if href.startswith("/"):
                href = urllib.parse.urljoin(url, href)

            # Validate URL
            if is_valid_url(href, base_domain, same_domain_only):
                links.append(href)

        return list(set(links))  # Remove duplicates
    except Exception as e:
        st.warning(f"Error extracting links from {url}: {e}")
        return []


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
        return data[0].page_content, url
    except Exception as e:
        st.warning(f"Error loading webpage {url}: {e}")
        return None, url


def crawl_website(start_url, config, progress_bar=None, status_text=None):
    """
    Crawl a website starting from a given URL up to a specified depth.

    Args:
        start_url (str): The starting URL for crawling.
        config (dict): Configuration parameters including crawl depth.
        progress_bar: Streamlit progress bar object.
        status_text: Streamlit text object for status updates.

    Returns:
        str: Combined text from all crawled pages.
    """
    base_domain = get_domain(start_url)
    to_crawl = [(start_url, 0)]  # (url, depth)
    crawled_urls = set()
    all_texts = []

    max_depth = config["crawl_depth"]
    max_pages = config["max_pages"]
    same_domain_only = config["same_domain_only"]

    if progress_bar:
        progress_bar.progress(0)

    while to_crawl and len(crawled_urls) < max_pages:
        current_url, depth = to_crawl.pop(0)

        # Skip if already crawled
        if current_url in crawled_urls:
            continue

        # Update status
        if status_text:
            status_text.text(
                f"Crawling page {len(crawled_urls) + 1}/{max_pages}: {current_url}"
            )

        # Extract text from current page
        result = extract_text_from_webpage(current_url)
        if result:
            text, url = result
            if text:
                all_texts.append(f"--- Content from {url} ---\n{text}")
                crawled_urls.add(current_url)
                st.session_state.crawled_urls.add(current_url)

        # If we haven't reached max depth, get links for next level
        if depth < max_depth:
            links = extract_links_from_page(current_url, base_domain, same_domain_only)
            # Add new links to crawl queue
            for link in links:
                if link not in crawled_urls and (link, depth + 1) not in to_crawl:
                    to_crawl.append((link, depth + 1))

        # Update progress
        if progress_bar:
            progress_percentage = min(len(crawled_urls) / max_pages, 1.0)
            progress_bar.progress(progress_percentage)

    # Final update
    if status_text:
        status_text.text(f"Crawling completed. Processed {len(crawled_urls)} pages.")

    # Return combined text from all pages
    return "\n\n".join(all_texts)


# ============= Text Processing Component =============
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
        You are an AI assistant that helps users understand website content.

        Website content:
        {text}

        User question: {user_input}

        Please provide a helpful, accurate, and concise answer based on the website content.
        If the answer is not in the content, say so clearly.
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
        response = qa_chain.invoke(
            {"question": user_input, "chat_history": st.session_state.chat_history}
        )
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
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational chain
    llm = ChatOllama(model=config["ollama_model"])
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    return qa_chain


def process_website_content(text, config):
    """
    Process website content by setting up a query processor.

    Args:
        text (str): The combined text from all crawled pages.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing:
            - function: The query processor function.
            - int: The length of the extracted text.
    """
    text_length = len(text)
    st.info(
        f"Processed {text_length} characters from {len(st.session_state.crawled_urls)} pages"
    )

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
    user_input = st.chat_input("Ask a question about the website:")

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
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


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
    url_input = st.text_input(
        "Enter a website URL to crawl:",
        "https://python.langchain.com/docs/get_started/introduction",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        crawl_button = st.button("Crawl Website")

    with col2:
        if st.session_state.crawled_urls:
            st.info(f"Crawled {len(st.session_state.crawled_urls)} pages")

    # Display crawled URLs in sidebar
    with st.sidebar:
        if st.session_state.crawled_urls:
            with st.expander("Crawled URLs"):
                for url in st.session_state.crawled_urls:
                    st.write(f"- {url}")

    if crawl_button and url_input and not st.session_state.crawl_in_progress:
        # Reset session state for new crawl
        st.session_state.crawled_urls = set()
        st.session_state.crawl_in_progress = True
        st.session_state.chat_history = []

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Crawl the website
            with st.spinner("Crawling website..."):
                combined_text = crawl_website(
                    url_input, config, progress_bar, status_text
                )

            if combined_text:
                # Process the crawled content
                query_processor, text_length = process_website_content(
                    combined_text, config
                )

                if query_processor is not None:
                    st.session_state.query_processor = query_processor
                    st.session_state.text_length = text_length
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": f"Website crawled successfully! Processed {len(st.session_state.crawled_urls)} pages. You can now ask questions about the content.",
                        }
                    ]
            else:
                st.error("No content was extracted from the website.")
        except Exception as e:
            st.error(f"Error during crawling: {e}")
        finally:
            st.session_state.crawl_in_progress = False
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

    # Display chat messages
    display_chat_messages()

    # Handle user input if a webpage has been processed
    if "query_processor" in st.session_state:
        handle_user_input(st.session_state.query_processor)


if __name__ == "__main__":
    main()

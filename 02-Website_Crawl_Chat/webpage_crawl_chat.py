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
from typing import TypedDict, Optional, List, Dict, Set
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor


# Define the state structure for LangGraph
class WebsiteCrawlerState(TypedDict):
    base_url: str
    max_depth: int
    max_pages: int
    crawled_pages: Dict[str, str]  # URL -> content
    urls_to_crawl: List[str]
    visited_urls: Set[str]
    current_depth: int
    approach: Optional[str]
    error: Optional[str]


# Function to extract text from a webpage
def extract_webpage_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
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

        return text, soup, None
    except Exception as e:
        return None, None, f"Error fetching webpage {url}: {str(e)}"


# Function to extract links from a webpage
def extract_links(soup, base_url):
    links = []
    if soup is None:
        return links

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Handle relative URLs
        full_url = urllib.parse.urljoin(base_url, href)
        # Filter out non-http links, anchors, etc.
        if full_url.startswith(('http://', 'https://')) and '#' not in full_url:
            # Stay within the same domain
            if urllib.parse.urlparse(full_url).netloc == urllib.parse.urlparse(base_url).netloc:
                links.append(full_url)

    return links


# Define LangGraph nodes
def initialize_crawler(state: WebsiteCrawlerState) -> WebsiteCrawlerState:
    """Initialize the crawler with the starting URL."""
    base_url = state["base_url"]

    # Initialize crawler state
    return {
        **state,
        "crawled_pages": {},
        "urls_to_crawl": [base_url],
        "visited_urls": set(),
        "current_depth": 0,
        "error": None
    }


def crawl_website(state: WebsiteCrawlerState) -> WebsiteCrawlerState:
    """Crawl the website to the specified depth."""
    max_depth = state["max_depth"]
    max_pages = state["max_pages"]
    base_url = state["base_url"]

    crawled_pages = state["crawled_pages"].copy()
    visited_urls = state["visited_urls"].copy()

    current_depth = 0
    urls_to_crawl = [base_url]

    # For progress tracking in Streamlit
    progress_placeholder = st.empty()

    while current_depth <= max_depth and urls_to_crawl and len(crawled_pages) < max_pages:
        next_urls = []

        # Update progress
        progress_placeholder.progress(min(len(crawled_pages) / max_pages, 1.0))
        progress_placeholder.text(f"Crawling: {len(crawled_pages)}/{max_pages} pages, depth {current_depth}/{max_depth}")

        # Process URLs at current depth (with concurrency for speed)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create a list of URLs to process at this depth (limited by max_pages)
            urls_batch = []
            for url in urls_to_crawl:
                if url not in visited_urls and len(urls_batch) + len(crawled_pages) < max_pages:
                    urls_batch.append(url)
                    visited_urls.add(url)

            # Process the batch concurrently
            for url, (content, soup, error) in zip(
                    urls_batch,
                    executor.map(lambda u: extract_webpage_content(u), urls_batch)
            ):
                if error is None and content:
                    crawled_pages[url] = content

                    # If we haven't reached max depth, collect links for next depth
                    if current_depth < max_depth:
                        links = extract_links(soup, url)
                        next_urls.extend([link for link in links if link not in visited_urls])

                # Small delay to be nice to servers
                time.sleep(0.1)

        # Move to next depth
        urls_to_crawl = next_urls
        current_depth += 1

    # Clear progress indicator
    progress_placeholder.empty()

    return {
        **state,
        "crawled_pages": crawled_pages,
        "visited_urls": visited_urls,
        "current_depth": current_depth,
        "error": None if crawled_pages else "No pages were successfully crawled"
    }


def determine_approach(state: WebsiteCrawlerState) -> WebsiteCrawlerState:
    """Determine whether to use full context or embeddings."""
    if state.get("error"):
        return state

    # Combine all crawled content
    all_content = "\n\n".join([
        f"--- {url} ---\n{content}"
        for url, content in state["crawled_pages"].items()
    ])

    threshold = st.session_state.get("text_threshold", 4000)
    approach = "full_context" if len(all_content) < threshold else "embeddings"

    return {**state, "approach": approach}


# Build the LangGraph workflow
def build_website_crawler():
    """Build a graph for crawling websites."""
    workflow = lg.StateGraph(WebsiteCrawlerState)

    # Add nodes to the graph
    workflow.add_node("initialize_crawler", initialize_crawler)
    workflow.add_node("crawl_website", crawl_website)
    workflow.add_node("determine_approach", determine_approach)

    # Add edges
    workflow.add_edge("initialize_crawler", "crawl_website")
    workflow.add_edge("crawl_website", "determine_approach")

    # Set entry point
    workflow.set_entry_point("initialize_crawler")

    # Compile the graph
    return workflow.compile()


# Function to set up full context chat
def setup_full_context_chat(crawled_pages, model_name):
    llm = Ollama(model=model_name)

    # Combine all content with page URLs as context
    all_content = "\n\n".join([
        f"--- {url} ---\n{content}"
        for url, content in crawled_pages.items()
    ])

    def get_response(query):
        prompt = f"""
        You are an AI assistant that helps users understand and extract information from websites.

        Website content from {len(crawled_pages)} pages:
        {all_content}

        Answer the user's question based on this website content. If the information is not in the content, say so.
        Include references to specific pages when relevant.

        User's question: {query}
        """
        return llm.invoke(prompt)

    return get_response


# Function to set up embeddings-based chat
def setup_embeddings_chat(crawled_pages, model_name):
    # Prepare documents with metadata
    documents = []

    for url, content in crawled_pages.items():
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        chunks = text_splitter.split_text(content)

        # Add source URL as metadata
        for chunk in chunks:
            documents.append({"content": chunk, "metadata": {"source": url}})

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=model_name)

    # Create FAISS index from documents
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    retriever = vectorstore.as_retriever()

    # Create memory for conversation history with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify which output to store in memory
    )

    # Create conversational chain
    llm = Ollama(model=model_name)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain


# Streamlit app
def main():
    st.title("Website Crawler and Chat with LangGraph and Ollama")

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
        max_value=50000,
        value=10000,
        step=1000,
        help="If combined website text exceeds this length, embeddings will be used instead of full context",
    )
    st.session_state.text_threshold = text_threshold

    # Crawler settings
    max_depth = st.sidebar.slider(
        "Maximum crawl depth",
        min_value=0,
        max_value=5,
        value=1,
        help="0 means only the initial page, 1 includes linked pages, etc."
    )

    max_pages = st.sidebar.slider(
        "Maximum pages to crawl",
        min_value=1,
        max_value=50,
        value=10,
        help="Limit the total number of pages to crawl"
    )

    # URL input
    url = st.text_input("Enter website URL:", "https://docs.streamlit.io/")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False

    # Process button
    if st.button("Crawl Website"):
        with st.spinner("Crawling website..."):
            # Set up the LangGraph workflow
            crawler = build_website_crawler()

            # Process the website using LangGraph
            result = crawler.invoke({
                "base_url": url,
                "max_depth": max_depth,
                "max_pages": max_pages,
                "crawled_pages": {},
                "urls_to_crawl": [],
                "visited_urls": set(),
                "current_depth": 0,
                "approach": None,
                "error": None
            })

            if result.get("error"):
                st.error(result["error"])
            else:
                # Reset chat
                st.session_state.messages = []

                # Store the crawled pages and approach
                crawled_pages = result["crawled_pages"]
                approach = result["approach"]

                st.session_state.crawled_pages = crawled_pages

                # Display crawl statistics
                st.success(f"Website crawled successfully! Processed {len(crawled_pages)} pages to depth {result['current_depth']}.")

                with st.expander("Crawled Pages"):
                    for url in crawled_pages.keys():
                        st.write(url)

                # Set up chat based on approach
                if approach == "full_context":
                    st.session_state.chat_engine = setup_full_context_chat(
                        crawled_pages, model_name
                    )
                else:
                    st.session_state.chat_engine = setup_embeddings_chat(
                        crawled_pages, model_name
                    )

                st.session_state.chat_initialized = True
                st.session_state.approach = approach
                st.info(f"Using {approach} approach for chat.")

    # Display chat interface if initialized
    if st.session_state.chat_initialized:
        st.subheader("Chat with the Website")
        st.info(f"Current approach: {st.session_state.approach}")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input for user query
        user_input = st.chat_input("Ask about the website:")

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

                        # Format response with sources
                        response = result["answer"]

                        # Add source references if available
                        if "source_documents" in result:
                            sources = set()
                            for doc in result["source_documents"]:
                                if "source" in doc.metadata:
                                    sources.add(doc.metadata["source"])

                            if sources:
                                response += "\n\nSources:\n" + "\n".join([f"- {src}" for src in sources])

                    # Display response
                    st.write(response)

                    # Add assistant response to chat
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )


if __name__ == "__main__":
    main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import langgraph.graph as lg
from typing import TypedDict, Optional, Dict, Set


# Define the state structure for LangGraph
class WebpageChatState(TypedDict):
    base_url: str
    crawl_depth: int
    enable_crawling: bool
    visited_urls: Set[str]
    pages_content: Dict[str, str]
    combined_content: Optional[str]
    approach: Optional[str]
    error: Optional[str]
    crawl_status: Optional[Dict]


# Function to check if a URL should be crawled
def is_valid_url_to_crawl(url, base_url):
    """Check if URL should be crawled based on various criteria."""
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)

    # Only crawl URLs from the same domain
    if parsed_url.netloc != parsed_base.netloc:
        return False

    # Skip URLs with fragments
    if parsed_url.fragment:
        return False

    # Skip certain file types
    skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js']
    if any(url.lower().endswith(ext) for ext in skip_extensions):
        return False

    return True


# Function to extract links from a webpage
def extract_links_from_page(url, html_content, base_url):
    """Extract valid links from an HTML page."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()

    for anchor in soup.find_all('a', href=True):
        link = anchor['href']
        absolute_link = urljoin(url, link)

        if is_valid_url_to_crawl(absolute_link, base_url):
            links.add(absolute_link)

    return links


# Function to extract text from a webpage
def extract_webpage_content(url):
    """Extract text content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
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

        return text, response.text, None
    except Exception as e:
        return None, None, f"Error fetching webpage {url}: {str(e)}"


# Function to crawl a website
def crawl_website(base_url, max_depth=2, max_pages=20):
    """Crawl a website up to a specified depth and max number of pages."""
    visited = set()
    to_visit = [(base_url, 0)]  # (url, depth)
    pages_content = {}
    error_logs = []

    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)

        if url in visited or depth > max_depth:
            continue

        visited.add(url)

        content, html, error = extract_webpage_content(url)
        if error:
            error_logs.append(error)
            continue

        # Store the content
        pages_content[url] = content

        # Don't extract more links if we've reached max depth
        if depth < max_depth:
            links = extract_links_from_page(url, html, base_url)
            for link in links:
                if link not in visited:
                    to_visit.append((link, depth + 1))

        # Short delay to be respectful to the server
        time.sleep(0.5)

    # Combine all content with URL markers for each section
    combined_content = ""
    for url, content in pages_content.items():
        combined_content += f"\n\n--- URL: {url} ---\n{content}"

    # Remove leading newlines if they exist
    combined_content = combined_content.lstrip("\n")

    return {
        "visited_urls": visited,
        "pages_content": pages_content,
        "combined_content": combined_content,
        "errors": error_logs
    }


# Define LangGraph nodes
def fetch_website(state: WebpageChatState) -> WebpageChatState:
    """Fetch content from a website, either single page or by crawling."""
    base_url = state["base_url"]

    if not state["enable_crawling"]:
        # Just fetch the single page
        content, _, error = extract_webpage_content(base_url)
        if error:
            return {**state, "error": error}

        return {
            **state,
            "visited_urls": {base_url},
            "pages_content": {base_url: content},
            "combined_content": content,
            "error": None
        }
    else:
        # Crawl the website
        crawl_result = crawl_website(
            base_url,
            max_depth=state["crawl_depth"],
            max_pages=st.session_state.get("max_pages", 20)
        )

        if not crawl_result["pages_content"]:
            return {**state, "error": "Failed to crawl any pages from the website."}

        return {
            **state,
            "visited_urls": crawl_result["visited_urls"],
            "pages_content": crawl_result["pages_content"],
            "combined_content": crawl_result["combined_content"],
            "crawl_status": {
                "pages_crawled": len(crawl_result["visited_urls"]),
                "errors": crawl_result["errors"]
            },
            "error": None
        }


def determine_approach(state: WebpageChatState) -> WebpageChatState:
    """Determine whether to use full context or embeddings."""
    if state.get("error"):
        return state

    content = state["combined_content"]
    threshold = st.session_state.get("text_threshold", 4000)

    approach = "full_context" if len(content) < threshold else "embeddings"

    return {**state, "approach": approach}


# Build the LangGraph workflow
def build_website_processor():
    """Build a graph for processing websites."""
    workflow = lg.StateGraph(WebpageChatState)

    # Add nodes to the graph
    workflow.add_node("fetch_website", fetch_website)
    workflow.add_node("determine_approach", determine_approach)

    # Add edges
    workflow.add_edge("fetch_website", "determine_approach")

    # Set entry point
    workflow.set_entry_point("fetch_website")

    # Compile the graph
    return workflow.compile()


# Function to set up full context chat
def setup_full_context_chat(content, model_name):
    """Set up a chat based on the full content of crawled pages."""
    llm = Ollama(model=model_name)

    def get_response(query):
        prompt = f"""
        You are an AI assistant that helps users understand and extract information from websites.

        Website content:
        {content}

        Answer the user's question based on this website content. If the information is not in the content, say so.
        If referring to specific information, mention which URL contains that information.

        User's question: {query}
        """
        return llm.invoke(prompt)

    return get_response


def setup_embeddings_chat(content, model_name):
    """Set up a retrieval-based chat using embeddings of crawled pages."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(content)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Create memory for conversation history with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # This tells memory which key to use from chain output
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
    st.title("Website Crawler & Chat with LangGraph and Ollama")

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
        help="If website text exceeds this length, embeddings will be used instead of full context",
    )
    st.session_state.text_threshold = text_threshold

    # Crawling settings
    st.sidebar.header("Crawling Settings")
    enable_crawling = st.sidebar.checkbox("Enable website crawling", value=True)

    crawl_depth = st.sidebar.slider(
        "Crawl depth",
        min_value=0,
        max_value=3,
        value=1,
        help="0: Only the base URL, 1: Base URL and direct links, etc."
    )

    max_pages = st.sidebar.slider(
        "Maximum pages to crawl",
        min_value=1,
        max_value=50,
        value=10
    )
    st.session_state.max_pages = max_pages

    # URL input
    url = st.text_input("Enter website URL:", "https://example.com")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False

    # Process button
    if st.button("Process Website"):
        with st.spinner("Processing website..."):
            # Set up the LangGraph workflow
            processor = build_website_processor()

            # Process the website using LangGraph
            result = processor.invoke({
                "base_url": url,
                "crawl_depth": crawl_depth,
                "enable_crawling": enable_crawling,
                "visited_urls": set(),
                "pages_content": {},
                "combined_content": None,
                "approach": None,
                "error": None,
                "crawl_status": None
            })

            if result.get("error"):
                st.error(result["error"])
            else:
                # Reset chat
                st.session_state.messages = []

                # Store the website content and approach
                content = result["combined_content"]
                approach = result["approach"]
                visited_urls = result["visited_urls"]

                # Display crawl info
                if enable_crawling and result.get("crawl_status"):
                    st.success(f"Crawled {len(visited_urls)} pages successfully!")

                    if result["crawl_status"].get("errors"):
                        st.warning(f"Encountered {len(result['crawl_status']['errors'])} errors during crawling.")
                        if st.checkbox("Show crawling errors"):
                            for error in result["crawl_status"]["errors"]:
                                st.error(error)

                # Set up chat based on approach
                if approach == "full_context":
                    st.session_state.chat_engine = setup_full_context_chat(
                        content, model_name
                    )
                else:  # embeddings approach
                    st.session_state.chat_engine = setup_embeddings_chat(
                        content, model_name
                    )

                st.session_state.chat_initialized = True
                st.session_state.approach = approach
                st.session_state.visited_urls = visited_urls

                st.success(
                    f"Website processed successfully! Using {approach} approach."
                )

    # Display crawled pages if available
    if st.session_state.get("chat_initialized") and st.session_state.get("visited_urls"):
        with st.expander("Show crawled pages"):
            st.write(f"Processed {len(st.session_state.visited_urls)} pages:")
            for url in st.session_state.visited_urls:
                st.write(f"- {url}")

    # Display chat interface if initialized
    if st.session_state.chat_initialized:
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
                        response = result["answer"]

                        # Include source documents if available
                        if "source_documents" in result:
                            source_docs = result.get("source_documents", [])
                            if source_docs:
                                response += "\n\nSources:\n"
                                seen_urls = set()
                                for doc in source_docs[:3]:  # Show top 3 sources
                                    text = doc.page_content
                                    # Extract URL from the content format we created
                                    if "--- URL:" in text:
                                        url_part = text.split("--- URL:")[1].split("---")[0].strip()
                                        if url_part not in seen_urls:
                                            response += f"- {url_part}\n"
                                            seen_urls.add(url_part)

                    # Display response
                    st.write(response)

                    # Add assistant response to chat
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )


if __name__ == "__main__":
    main()
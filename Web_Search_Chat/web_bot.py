# llm_chatbot_app.py

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
import re  # Import regex for fallback URL extraction
import warnings  # Import the warnings module

# --- Suppress the specific UserWarning from Langchain ---
# This is a workaround because setting output_key might not work reliably in older versions
# The recommended fix is to upgrade the langchain library (`pip install --upgrade langchain`)
warnings.filterwarnings(
    "ignore",
    message=".*'ConversationBufferMemory' got multiple output keys.*",  # Match the warning message pattern
    category=UserWarning,
    module="langchain.memory.chat_memory",  # Be specific about the source
)
# --- End Warning Suppression ---


# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"
AGENT_PROMPT_REPO = "hwchase17/react-chat"  # A reliable ReAct chat prompt

# --- Initialization ---

# 1. Initialize LLM
# Use a flag in session state to avoid re-checking LLM status unnecessarily on every rerun
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if not st.session_state.llm_initialized:
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
        # Simple test call
        llm.invoke("Hello!")
        st.session_state.llm = llm  # Store llm in session state if successful
        st.session_state.llm_initialized = True
    except Exception as e:
        st.error(f"Error connecting to Ollama or model '{OLLAMA_MODEL}': {e}")
        st.error(
            f"Please ensure the Ollama server is running at {OLLAMA_BASE_URL} and the model '{OLLAMA_MODEL}' is pulled."
        )
        st.stop()  # Stop execution if LLM isn't available on first run
# On subsequent reruns, retrieve the llm from session state
llm = st.session_state.get("llm", None)
if not llm:  # Should not happen if initialization logic is correct, but as a safeguard
    st.error("LLM not found in session state. Please refresh.")
    st.stop()


# 2. Initialize Tools
# Using DuckDuckGoSearchResults to get URLs
search_tool = DuckDuckGoSearchResults(name="duckduckgo_search_results")
tools = [search_tool]

# 3. Initialize Memory
# Use Streamlit session state for robust memory handling across reruns
# NOTE: We are NOT adding output_key='output' here because it didn't work in the user's old version.
# The warning is being suppressed by warnings.filterwarnings above.
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True  # Return actual message objects
    )
# Keep a separate list for displaying messages in Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Initialize Agent
# Pull the ReAct chat prompt
prompt = hub.pull(AGENT_PROMPT_REPO)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor only if it doesn't exist in session state
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=True,  # Set to True to see agent's thought process in console
        handle_parsing_errors=True,  # Gracefully handle potential LLM output parsing errors
        return_intermediate_steps=True,  # Crucial for accessing tool usage and outputs
    )
# Retrieve agent executor from session state
agent_executor = st.session_state.agent_executor


# --- Streamlit UI ---

st.set_page_config(page_title=f"Chat with {OLLAMA_MODEL}", layout="wide")
st.title(f"🤖 Chatbot powered by Ollama ({OLLAMA_MODEL}) & Langchain")
st.caption("This chatbot can use DuckDuckGo to search the web for recent information.")

# Display chat messages from history
for message_info in st.session_state.messages:
    with st.chat_message(message_info["role"]):
        st.markdown(message_info["content"])
        # Display sources if they exist for an assistant message
        if "sources" in message_info and message_info["sources"]:
            st.caption("Sources:")
            # Ensure sources are displayed correctly even after rerun
            processed_sources = []
            for source in message_info["sources"]:
                if isinstance(source, dict) and "link" in source:
                    processed_sources.append(
                        f"- [{source.get('title', 'Source Link')}]({source['link']})"
                    )
                elif isinstance(source, str) and source.startswith("http"):
                    processed_sources.append(f"- {source}")
            st.markdown("\n".join(processed_sources), unsafe_allow_html=True)


# Accept user input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare input for the agent
    agent_input = {"input": prompt}

    # Display thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent
                response = agent_executor.invoke(agent_input)

                # Extract the final answer
                agent_response_content = response.get(
                    "output", "Sorry, I couldn't process that."
                )

                # Extract source information from intermediate steps
                source_links = set()  # Use a set to avoid duplicate links
                source_details = []  # Store dicts with link and title if available
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        action, observation = step
                        # Check if the action used the search tool (more robust check)
                        if hasattr(action, "tool") and action.tool == search_tool.name:
                            # Observation from DuckDuckGoSearchResults should be a list of dicts or string
                            if isinstance(observation, list):
                                for result in observation:
                                    if isinstance(result, dict) and "link" in result:
                                        link = result["link"]
                                        if link not in source_links:
                                            source_links.add(link)
                                            source_details.append(
                                                {
                                                    "link": link,
                                                    "title": result.get(
                                                        "title", "Source Link"
                                                    ),  # Get title if available
                                                }
                                            )
                            elif isinstance(observation, str):
                                # Basic check for URLs in the string observation as a fallback
                                found_urls = re.findall(
                                    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                    observation,
                                )
                                for url in found_urls:
                                    if url not in source_links:
                                        source_links.add(url)
                                        source_details.append(
                                            {"link": url, "title": "Source Link"}
                                        )

                # Display the agent's final response
                st.markdown(agent_response_content)

                # Display collected sources if any were found
                if source_details:
                    st.caption("Sources:")
                    source_markdown = []
                    for detail in sorted(
                        source_details, key=lambda x: x["link"]
                    ):  # Sort for consistent display
                        source_markdown.append(
                            f"- [{detail['title']}]({detail['link']})"
                        )
                    st.markdown("\n".join(source_markdown), unsafe_allow_html=True)

                # Add assistant response (and sources) to session state for display history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": agent_response_content,
                        # Store detailed sources for better display later
                        "sources": source_details,
                    }
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

# Optional: Add a button to clear history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    # Re-initialize memory if clearing
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # Optionally clear agent executor if its state depends heavily on memory nuances not captured by clearing memory alone
    if "agent_executor" in st.session_state:
        del st.session_state["agent_executor"]
    st.rerun()  # Rerun the app to reflect the cleared state

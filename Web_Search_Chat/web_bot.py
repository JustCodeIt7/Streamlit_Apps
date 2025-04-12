# llm_chatbot_app.py

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"
AGENT_PROMPT_REPO = "hwchase17/react-chat"  # A reliable ReAct chat prompt

# --- Initialization ---

# 1. Initialize LLM
# Check if Ollama server is available and the model exists
try:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
    # Simple test call
    llm.invoke("Hello!")
except Exception as e:
    st.error(f"Error connecting to Ollama or model '{OLLAMA_MODEL}': {e}")
    st.error(
        f"Please ensure the Ollama server is running at {OLLAMA_BASE_URL} and the model '{OLLAMA_MODEL}' is pulled."
    )
    st.stop()  # Stop execution if LLM isn't available

# 2. Initialize Tools
# Using DuckDuckGoSearchResults to get URLs
search_tool = DuckDuckGoSearchResults(name="duckduckgo_search_results")
tools = [search_tool]

# 3. Initialize Memory
# Use Streamlit session state for robust memory handling across reruns
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

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=st.session_state.memory,
    verbose=True,  # Set to True to see agent's thought process in console
    handle_parsing_errors=True,  # Gracefully handle potential LLM output parsing errors
    return_intermediate_steps=True,  # Crucial for accessing tool usage and outputs
)

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
            for source in message_info["sources"]:
                # Use a more robust way to display potential source links
                if isinstance(source, dict) and "link" in source:
                    st.caption(
                        f"- [{source.get('title', 'Source Link')}]({source['link']})"
                    )
                elif isinstance(source, str) and source.startswith("http"):
                    st.caption(f"- {source}")
                # Add more checks if needed based on actual tool output format


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
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        action, observation = step
                        # Check if the action used the search tool
                        if action.tool == search_tool.name:
                            # Observation from DuckDuckGoSearchResults should be a list of dicts
                            if isinstance(observation, list):
                                for result in observation:
                                    if isinstance(result, dict) and "link" in result:
                                        source_links.add(result["link"])
                            elif isinstance(observation, str):
                                # Sometimes it might return a string summary; less ideal for links
                                # Basic check for URLs in the string observation as a fallback
                                import re

                                found_urls = re.findall(
                                    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                    observation,
                                )
                                for url in found_urls:
                                    source_links.add(url)

                # Display the agent's final response
                st.markdown(agent_response_content)

                # Display collected sources if any were found
                if source_links:
                    st.caption("Sources:")
                    for link in sorted(
                        list(source_links)
                    ):  # Sort for consistent display
                        st.caption(f"- {link}")

                # Add assistant response (and sources) to session state for display history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": agent_response_content,
                        # Store sources as a list for potential reuse/display formatting
                        "sources": [
                            {"link": link, "title": "Source Link"}
                            for link in source_links
                        ],
                    }
                )

                # Note: The ConversationBufferMemory (`st.session_state.memory`)
                # is automatically updated by the AgentExecutor. We don't need
                # to manually add messages there *if* we use the memory object
                # directly within the executor. We are manually managing
                # st.session_state.messages purely for the UI display.

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

# Optional: Add a button to clear history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory.clear()  # Clear Langchain memory too
    st.rerun()  # Rerun the app to reflect the cleared state

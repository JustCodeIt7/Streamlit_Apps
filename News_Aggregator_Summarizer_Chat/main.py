import streamlit as st
import requests
import openai
import os

# -------------------------------------------------------------------
# API Key Setup
# -------------------------------------------------------------------
# You can set your API keys via environment variables or using Streamlit secrets.
# Example (in terminal):
#   export NEWS_API_KEY="your_newsapi_key_here"
#   export OPENAI_API_KEY="your_openai_api_key_here"

NEWS_API_KEY = st.secrets.get("news_api_key", os.getenv("NEWS_API_KEY"))
OPENAI_API_KEY = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

if not NEWS_API_KEY:
    st.error("Missing News API key! Set your NEWS_API_KEY in the environment or in Streamlit secrets.")
    st.stop()
if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key! Set your OPENAI_API_KEY in the environment or in Streamlit secrets.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def fetch_news(query=None, category="general", country="us", page_size=10):
    """
    Fetch news articles from NewsAPI based on the provided parameters.
    """
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "country": country,
        "category": category,
        "pageSize": page_size
    }
    if query and query.strip():
        params["q"] = query.strip()
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") != "ok":
        st.error("Error fetching news: " + data.get("message", "Unknown error"))
        return []
    return data.get("articles", [])

def generate_answer(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
    """
    Generate an answer using OpenAI's ChatCompletion API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------
def main():
    st.title("News Aggregator and Summarizer")
    st.write(
        """
        Aggregate the latest news articles or blog posts and ask for summaries or deep dives on specific topics.
        You can also request sentiment analysis or trend spotting.
        """
    )

    # ---------------------------
    # Sidebar: News Filters
    # ---------------------------
    st.sidebar.header("News Filters")
    country = st.sidebar.selectbox("Country", options=["us", "gb", "ca", "au", "in"], index=0)
    category = st.sidebar.selectbox(
        "Category",
        options=["general", "business", "entertainment", "health", "science", "sports", "technology"],
        index=0
    )
    query = st.sidebar.text_input("Search Query (optional)", "")
    page_size = st.sidebar.number_input("Number of articles", min_value=1, max_value=50, value=10)

    if st.sidebar.button("Fetch News"):
        with st.spinner("Fetching news..."):
            articles = fetch_news(query=query, category=category, country=country, page_size=page_size)
            st.session_state.articles = articles
            st.success(f"Fetched {len(articles)} articles.")

    # ---------------------------
    # Display Fetched News
    # ---------------------------
    if "articles" in st.session_state and st.session_state.articles:
        st.header("Latest News")
        for i, article in enumerate(st.session_state.articles):
            st.subheader(f"{i+1}. {article.get('title')}")
            st.write(article.get("description"))
            if article.get("url"):
                st.markdown(f"[Read more]({article.get('url')})")
            st.write("---")
    else:
        st.info("No news articles loaded yet. Use the sidebar to fetch the latest news.")

    # ---------------------------
    # Chat with the News Articles
    # ---------------------------
    st.header("Chat with the News")
    st.write(
        """
        Ask a question about the news articles above. For example:
        - "Summarize the main points of the top article."
        - "What is the overall sentiment of these news stories?"
        - "Identify any trends or patterns in the recent news."
        """
    )
    user_question = st.text_input("Enter your question about the news:")
    analysis_option = st.selectbox("Choose analysis type:", ["General", "Sentiment Analysis", "Trend Spotting"])

    if st.button("Get Answer") and user_question:
        # Prepare context from the fetched articles
        if "articles" in st.session_state and st.session_state.articles:
            # Combine title and description for context
            news_context = "\n\n".join(
                [f"Title: {a.get('title')}\nDescription: {a.get('description')}" for a in st.session_state.articles if a.get('title')]
            )
        else:
            news_context = "No news articles available."

        # Construct the prompt for the LLM
        prompt = (
            f"Given the following news articles:\n\n{news_context}\n\n"
            f"User question: {user_question}\n\n"
        )
        if analysis_option == "General":
            prompt += "Provide a comprehensive answer based on the news articles above."
        elif analysis_option == "Sentiment Analysis":
            prompt += "Analyze the sentiment expressed in these news articles and provide insights."
        elif analysis_option == "Trend Spotting":
            prompt += "Identify any trends or patterns in these news articles and elaborate on them."

        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt)
        st.subheader("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()

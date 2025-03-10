import streamlit as st
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="News Aggregator and Summarizer", page_icon="📰", layout="wide"
)


# Initialize NLTK components
@st.cache_resource
def load_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    return True


load_nltk_successful = load_nltk_resources()

# Define news sources (RSS feeds)
NEWS_SOURCES = {
    "CNN": "http://rss.cnn.com/rss/cnn_topstories.rss",
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
}


# Function to fetch news articles
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_news_from_rss(sources=None, max_articles=20):
    if sources is None or len(sources) == 0:
        sources = list(NEWS_SOURCES.keys())[:3]

    articles = []

    for source in sources:
        if source in NEWS_SOURCES:
            try:
                feed = feedparser.parse(NEWS_SOURCES[source])

                for entry in feed.entries[: max_articles // len(sources)]:
                    # Publication date
                    published = entry.get("published", entry.get("pubDate", "Unknown"))

                    # Extract image
                    image_url = None
                    if "media_content" in entry:
                        for media in entry.media_content:
                            if "url" in media:
                                image_url = media["url"]
                                break

                    # Extract content
                    summary = entry.get("summary", "")
                    description = entry.get("description", "")

                    # Clean up HTML tags
                    summary = re.sub("<.*?>", "", summary)
                    description = re.sub("<.*?>", "", description)

                    article = {
                        "title": entry.title,
                        "source": source,
                        "link": entry.link,
                        "published": published,
                        "summary": summary if summary else description,
                        "image": image_url,
                    }

                    articles.append(article)
            except Exception:
                st.sidebar.error(f"Error fetching news from {source}")

    return articles


# Sentiment analysis
def analyze_sentiment(text):
    if not text or not load_nltk_successful:
        return None

    try:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)
    except:
        return None


# Search articles by query
def search_articles(articles, query):
    if not query:
        return articles

    query = query.lower()
    return [
        article
        for article in articles
        if query in article["title"].lower()
        or query in article.get("summary", "").lower()
    ]


# Extract trending topics
def extract_trending_topics(articles, n=5):
    if not articles or not load_nltk_successful:
        return []

    try:
        all_text = ""
        for article in articles:
            all_text += article["title"] + " " + article.get("summary", "") + " "

        words = re.findall(r"\b[A-Za-z][A-Za-z]{3,}\b", all_text)

        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))

        additional_stop_words = {
            "said",
            "says",
            "news",
            "read",
            "reuters",
            "cnn",
            "bbc",
            "npr",
            "new",
            "now",
            "today",
            "latest",
        }
        stop_words.update(additional_stop_words)

        filtered_words = [
            word.lower() for word in words if word.lower() not in stop_words
        ]
        word_counts = Counter(filtered_words).most_common(n)

        return [word for word, count in word_counts]
    except:
        return []


# Generate summaries
def generate_summary(articles, topic=None):
    if not articles:
        return []

    if topic:
        # Filter articles related to the topic
        topic = topic.lower()
        filtered_articles = [
            article
            for article in articles
            if topic in article["title"].lower()
            or topic in article.get("summary", "").lower()
        ]
        target_articles = filtered_articles[:3]
    else:
        target_articles = articles[:3]

    summaries = []
    for article in target_articles:
        summary = article.get("summary", "")
        if summary:
            summary_text = f"**{article['title']}** (via {article['source']})\n\n{summary[:200]}..."
            summary_text += f"\n\n[Read full article]({article['link']})"
            summaries.append(summary_text)

    return summaries


# Answer user questions
def answer_question(question, articles):
    if not articles:
        return "Please refresh the news first to load articles."

    question = question.lower()

    # Handle summarization requests
    if any(term in question for term in ["summarize", "summary", "brief"]):
        summaries = generate_summary(articles)
        if summaries:
            return (
                "Here are summaries of the latest articles:\n\n"
                + "\n\n---\n\n".join(summaries)
            )
        return "No articles are available to summarize."

    # Handle sentiment analysis requests
    elif any(term in question for term in ["sentiment", "feeling", "tone", "mood"]):
        sentiments = [
            analyze_sentiment(article["title"] + " " + article.get("summary", ""))
            for article in articles
        ]
        sentiments = [s for s in sentiments if s]

        if not sentiments:
            return "No sentiment data is available for the current articles."

        positive_count = sum(1 for s in sentiments if s["compound"] >= 0.05)
        negative_count = sum(1 for s in sentiments if s["compound"] <= -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count
        total = len(sentiments)

        response = f"Based on analysis of {total} articles:\n"
        response += f"- {positive_count} articles have a positive tone ({positive_count/total*100:.1f}%)\n"
        response += f"- {negative_count} articles have a negative tone ({negative_count/total*100:.1f}%)\n"
        response += f"- {neutral_count} articles have a neutral tone ({neutral_count/total*100:.1f}%)\n\n"

        if positive_count > negative_count and positive_count > neutral_count:
            response += "The overall sentiment of recent news is **positive**."
        elif negative_count > positive_count and negative_count > neutral_count:
            response += "The overall sentiment of recent news is **negative**."
        else:
            response += "The overall sentiment of recent news is **neutral**."

        return response

    # Handle trending topics requests
    elif any(term in question for term in ["trending", "trend", "popular", "topics"]):
        trending = extract_trending_topics(articles)
        if trending:
            return "**Current trending topics:**\n" + "\n".join(
                f"{i+1}. {topic.capitalize()}" for i, topic in enumerate(trending)
            )
        return "No trending topics could be identified from the current articles."

    # Handle specific topic inquiries
    else:
        words = re.findall(r"\b\w+\b", question)
        words = [word for word in words if len(word) > 3]

        stop_words = [
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "tell",
            "about",
            "regarding",
            "concerning",
            "articles",
            "news",
        ]
        potential_topics = [word for word in words if word not in stop_words]

        if potential_topics:
            topic = potential_topics[0]
            summaries = generate_summary(articles, topic)

            if summaries:
                return (
                    f"Here are articles related to '{topic}':\n\n"
                    + "\n\n---\n\n".join(summaries)
                )

        return (
            "I can help you with: \n"
            "- Summarizing recent news articles\n"
            "- Analyzing the sentiment of current news\n"
            "- Identifying trending topics\n"
            "- Finding articles on specific topics\n\n"
            "Try asking something like 'Summarize the recent news' or 'What's trending right now?'"
        )


# Main app
def main():
    st.title("📰 News Aggregator and Summarizer")

    # Initialize session state
    if "articles" not in st.session_state:
        st.session_state.articles = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None

    # Create layout columns
    col1, col2 = st.columns([2, 3])

    with col1:
        # Source selection
        st.subheader("News Sources")
        selected_sources = st.multiselect(
            "Select news sources",
            options=list(NEWS_SOURCES.keys()),
            default=list(NEWS_SOURCES.keys())[:2],
        )

        # Search functionality
        search_query = st.text_input("Search articles")

        # Refresh button
        if st.button("Refresh News"):
            with st.spinner("Fetching latest news..."):
                st.session_state.articles = fetch_news_from_rss(
                    sources=selected_sources, max_articles=20
                )
                st.session_state.last_refresh = datetime.now()
                st.success("News updated!")

        if st.session_state.last_refresh:
            st.caption(
                f"Last updated: {st.session_state.last_refresh.strftime('%H:%M')}"
            )

        # Display trending topics
        if st.session_state.articles:
            trending = extract_trending_topics(st.session_state.articles)
            if trending:
                st.subheader("Trending Topics")
                for topic in trending:
                    st.write(f"#{topic.capitalize()}")

        # Chat section
        st.subheader("Chat with the News")

        # Display chat history
        chat_container = st.container(height=300)
        with chat_container:
            for message in st.session_state.chat_history:
                st.markdown(f"**{message['role'].title()}:** {message['content']}")

        # User input
        user_query = st.text_input("Your question:", key="chat_input")

        if st.button("Ask") and user_query:
            # Add user query to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_query}
            )

            # Generate response
            response = answer_question(user_query, st.session_state.articles)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

            # Rerun to update UI
            st.experimental_rerun()

    with col2:
        # Filter articles based on search
        display_articles = st.session_state.articles
        if search_query:
            display_articles = search_articles(display_articles, search_query)
            st.caption(
                f"Found {len(display_articles)} articles matching '{search_query}'"
            )

        # Display articles
        if not display_articles:
            st.info(
                "No articles to display. Please select news sources and click 'Refresh News'."
            )
        else:
            # Create tabs for viewing options
            tab1, tab2 = st.tabs(["Articles", "Analytics"])

            with tab1:
                # Display articles
                for i, article in enumerate(display_articles):
                    with st.expander(
                        f"{i+1}. {article['title']} - {article['source']}"
                    ):
                        st.caption(f"Published: {article['published']}")

                        if article.get("image"):
                            st.image(article["image"], use_column_width=True)

                        st.markdown("**Summary:**")
                        st.write(article.get("summary", "No summary available"))

                        # Sentiment analysis
                        sentiment = analyze_sentiment(
                            article["title"] + " " + article.get("summary", "")
                        )
                        if sentiment:
                            st.markdown("**Sentiment:**")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Positive", f"{sentiment['pos']:.2f}")
                            col2.metric("Neutral", f"{sentiment['neu']:.2f}")
                            col3.metric("Negative", f"{sentiment['neg']:.2f}")

                        st.markdown(f"[Read full article]({article['link']})")

            with tab2:
                # Overall sentiment analysis
                st.subheader("Sentiment Analysis of All Articles")

                sentiments = [
                    analyze_sentiment(
                        article["title"] + " " + article.get("summary", "")
                    )
                    for article in display_articles
                ]
                sentiments = [s for s in sentiments if s]

                if sentiments:
                    # Create summary of sentiments
                    positive = sum(1 for s in sentiments if s["compound"] >= 0.05)
                    negative = sum(1 for s in sentiments if s["compound"] <= -0.05)
                    neutral = len(sentiments) - positive - negative

                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(
                        [positive, negative, neutral],
                        labels=["Positive", "Negative", "Neutral"],
                        autopct="%1.1f%%",
                        colors=["green", "red", "gray"],
                    )
                    ax.set_title("Sentiment Distribution")
                    st.pyplot(fig)

                    # Display trending topics visualization
                    st.subheader("Trending Topics")
                    trending = extract_trending_topics(display_articles, n=8)

                    if trending:
                        topic_df = pd.DataFrame(
                            {"Topic": trending, "Count": range(len(trending), 0, -1)}
                        )

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(topic_df["Topic"], topic_df["Count"], color="skyblue")
                        ax.set_title("Trending Topics")
                        st.pyplot(fig)
                else:
                    st.info("No sentiment data available for analysis.")


if __name__ == "__main__":
    main()

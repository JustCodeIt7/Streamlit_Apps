import streamlit as st
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Initialize NLTK components
try:
    nltk.download("punkt", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
except:
    pass

# Define news sources (RSS feeds)
NEWS_SOURCES = {
    "CNN": "http://rss.cnn.com/rss/cnn_topstories.rss",
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "Fox News": "http://feeds.foxnews.com/foxnews/latest",
}


# Function to fetch articles from RSS feeds
def fetch_news_from_rss(sources=None, max_articles=10):
    if sources is None or len(sources) == 0:
        sources = list(NEWS_SOURCES.keys())[:3]  # Default to first 3 sources

    articles = []

    for source in sources:
        if source in NEWS_SOURCES:
            try:
                feed = feedparser.parse(NEWS_SOURCES[source])

                for entry in feed.entries[: max_articles // len(sources)]:
                    # Extract publication date
                    if "published" in entry:
                        published = entry.published
                    elif "pubDate" in entry:
                        published = entry.pubDate
                    else:
                        published = "Unknown"

                    # Extract image if available
                    image_url = None
                    if "media_content" in entry:
                        for media in entry.media_content:
                            if "url" in media:
                                image_url = media["url"]
                                break

                    # Create article dictionary
                    article = {
                        "title": entry.title,
                        "source": source,
                        "link": entry.link,
                        "published": published,
                        "summary": entry.get("summary", ""),
                        "description": entry.get("description", ""),
                        "image": image_url,
                    }

                    articles.append(article)
            except Exception as e:
                st.error(f"Error fetching news from {source}: {e}")

    return articles


# Function to perform sentiment analysis
def analyze_sentiment(text):
    if not text:
        return None

    try:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)
    except:
        return None


# Function to search for articles based on query
def search_articles(articles, query):
    if not query:
        return articles

    query = query.lower()
    results = []

    for article in articles:
        # Search in title and summary
        title = article["title"].lower()
        summary = article.get("summary", "").lower()
        description = article.get("description", "").lower()

        if (query in title) or (query in summary) or (query in description):
            results.append(article)

    return results


# Function to extract common words for trending topics
def extract_trending_topics(articles, n=5):
    # Combine all text
    all_text = ""

    for article in articles:
        all_text += article["title"] + " "
        all_text += article.get("summary", "") + " "
        all_text += article.get("description", "") + " "

    # Extract words
    words = re.findall(r"\b[A-Za-z][A-Za-z]{3,}\b", all_text)

    # Remove common English words
    common_words = {
        "the",
        "and",
        "that",
        "have",
        "for",
        "not",
        "you",
        "with",
        "this",
        "but",
        "his",
        "from",
        "they",
        "who",
        "say",
        "will",
        "what",
        "make",
        "when",
        "can",
        "more",
        "been",
        "their",
        "also",
        "would",
        "about",
        "news",
        "said",
        "just",
    }

    filtered_words = [
        word.lower() for word in words if word.lower() not in common_words
    ]

    # Count frequency
    word_counts = pd.Series(filtered_words).value_counts()

    # Get top N trending topics
    trending = word_counts.head(n).index.tolist()

    return trending


# Function to generate simple summaries
def generate_summary(articles, topic=None):
    summaries = []

    if topic:
        # Filter articles related to the topic
        filtered_articles = []
        topic = topic.lower()

        for article in articles:
            if (
                topic in article["title"].lower()
                or topic in article.get("summary", "").lower()
                or topic in article.get("description", "").lower()
            ):
                filtered_articles.append(article)

        target_articles = filtered_articles[:3]  # Limit to 3 related articles
    else:
        target_articles = articles[:3]  # Just take the first 3 articles

    for article in target_articles:
        title = article["title"]
        summary = article.get("summary", article.get("description", ""))

        if summary:
            # Clean up HTML tags if present
            summary = re.sub("<.*?>", "", summary)
            summaries.append(f"**{title}**: {summary}")

    return summaries


# Function to answer questions about the articles
def answer_question(question, articles):
    question = question.lower()

    # Check for summarization requests
    if any(
        term in question for term in ["summarize", "summary", "summarization", "brief"]
    ):
        summaries = generate_summary(articles)

        if summaries:
            response = "Here are summaries of the latest articles:\n\n"
            response += "\n\n".join(summaries)
        else:
            response = "No articles are available to summarize."

        return response

    # Check for sentiment analysis requests
    elif any(term in question for term in ["sentiment", "feeling", "tone", "mood"]):
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            # Combine title and summary for sentiment analysis
            text = article["title"] + " " + article.get("summary", "")
            sentiment = analyze_sentiment(text)

            if sentiment:
                compound = sentiment["compound"]

                if compound >= 0.05:
                    positive_count += 1
                elif compound <= -0.05:
                    negative_count += 1
                else:
                    neutral_count += 1

        total = positive_count + negative_count + neutral_count

        if total == 0:
            return "No sentiment data is available for the current articles."

        response = f"Based on analysis of {total} articles:\n"
        response += f"- {positive_count} articles have a positive tone ({positive_count/total*100:.1f}%)\n"
        response += f"- {negative_count} articles have a negative tone ({negative_count/total*100:.1f}%)\n"
        response += f"- {neutral_count} articles have a neutral tone ({neutral_count/total*100:.1f}%)\n\n"

        # Determine overall sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            response += "The overall sentiment of recent news is **positive**."
        elif negative_count > positive_count and negative_count > neutral_count:
            response += "The overall sentiment of recent news is **negative**."
        else:
            response += "The overall sentiment of recent news is **neutral**."

        return response

    # Check for trending topics requests
    elif any(
        term in question for term in ["trending", "trend", "popular", "hot topics"]
    ):
        trending = extract_trending_topics(articles)

        if trending:
            response = "**Current trending topics:**\n"
            for i, topic in enumerate(trending):
                response += f"{i+1}. {topic.capitalize()}\n"

            return response
        else:
            return "No trending topics could be identified from the current articles."

    # Check for specific topic inquiries
    else:
        # Extract potential topics from the question
        words = re.findall(r"\b\w+\b", question)
        words = [word for word in words if len(word) > 3]  # Filter out short words

        # Remove common question words
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
            topic = potential_topics[0]  # Take the first potential topic

            # Search for articles related to this topic
            related_articles = []

            for article in articles:
                if (
                    topic in article["title"].lower()
                    or topic in article.get("summary", "").lower()
                    or topic in article.get("description", "").lower()
                ):
                    related_articles.append(article)

            if related_articles:
                summaries = generate_summary(related_articles)

                if summaries:
                    response = f"Here are articles related to '{topic}':\n\n"
                    response += "\n\n".join(summaries)
                    return response

        # Default response if no specific topic is identified
        return (
            "I can help you with: \n"
            "- Summarizing recent news articles\n"
            "- Analyzing the sentiment of current news\n"
            "- Identifying trending topics\n"
            "- Finding articles on specific topics\n\n"
            "Try asking something like 'Summarize the recent news' or 'What's trending right now?'"
        )


# Main Streamlit app
def main():
    st.title("News Aggregator and Summarizer")

    # Initialize session state
    if "articles" not in st.session_state:
        st.session_state.articles = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None

    # Sidebar - News Sources Selection
    st.sidebar.title("News Sources")

    selected_sources = st.sidebar.multiselect(
        "Select news sources",
        options=list(NEWS_SOURCES.keys()),
        default=list(NEWS_SOURCES.keys())[:3],
    )

    # Sidebar - Search
    search_query = st.sidebar.text_input("Search articles")

    # Refresh button
    col1, col2 = st.sidebar.columns([1, 1])

    with col1:
        if st.button("Refresh News"):
            with st.spinner("Fetching latest news..."):
                # Fetch articles from RSS feeds
                st.session_state.articles = fetch_news_from_rss(
                    sources=selected_sources, max_articles=20
                )
                st.session_state.last_refresh = datetime.now()

    with col2:
        if st.session_state.last_refresh:
            st.write(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M')}")

    # Display trending topics if available
    if st.session_state.articles:
        trending = extract_trending_topics(st.session_state.articles)

        if trending:
            st.sidebar.title("Trending Topics")
            for topic in trending:
                if st.sidebar.button(topic.capitalize()):
                    search_query = topic

    # Filter articles based on search query
    display_articles = st.session_state.articles

    if search_query:
        display_articles = search_articles(display_articles, search_query)
        st.write(f"Found {len(display_articles)} articles matching '{search_query}'")

    # Main content area - Articles
    if not display_articles:
        st.info(
            "No articles to display. Please select news sources and click 'Refresh News'."
        )
    else:
        # Create tabs for viewing options
        tab1, tab2 = st.tabs(["Articles", "Analytics"])

        with tab1:
            # Display articles in expandable sections
            for i, article in enumerate(display_articles):
                with st.expander(f"{i+1}. {article['title']} - {article['source']}"):
                    st.write(f"**Published:** {article['published']}")

                    # Display image if available
                    if "image" in article and article["image"]:
                        st.image(article["image"], use_column_width=True)

                    # Display summary/description
                    summary = article.get(
                        "summary", article.get("description", "No summary available")
                    )
                    # Clean up HTML tags if present
                    summary = re.sub("<.*?>", "", summary)
                    st.write("**Summary:**")
                    st.write(summary)

                    # Perform and display sentiment analysis
                    sentiment = analyze_sentiment(article["title"] + " " + summary)
                    if sentiment:
                        st.write("**Sentiment Analysis:**")

                        # Create sentiment visualization
                        fig, ax = plt.subplots(figsize=(10, 2))
                        sentiments = ["Negative", "Neutral", "Positive"]
                        values = [sentiment["neg"], sentiment["neu"], sentiment["pos"]]

                        ax.barh(sentiments, values, color=["red", "gray", "green"])
                        ax.set_xlim(0, 1)
                        ax.set_title("Sentiment Analysis")

                        st.pyplot(fig)

                    # Link to full article
                    st.write(f"[Read full article]({article['link']})")

        with tab2:
            # Overall sentiment analysis
            st.subheader("Sentiment Analysis of All Articles")

            # Calculate overall sentiment
            sentiments = []
            for article in display_articles:
                text = article["title"] + " " + article.get("summary", "")
                sentiment = analyze_sentiment(text)
                if sentiment:
                    sentiments.append(sentiment)

            if sentiments:
                # Create summary of sentiments
                positive = sum(1 for s in sentiments if s["compound"] >= 0.05)
                negative = sum(1 for s in sentiments if s["compound"] <= -0.05)
                neutral = len(sentiments) - positive - negative

                # Create a pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(
                    [positive, negative, neutral],
                    labels=["Positive", "Negative", "Neutral"],
                    autopct="%1.1f%%",
                    colors=["green", "red", "gray"],
                )
                ax.set_title("Overall Sentiment Distribution")
                st.pyplot(fig)

                # Display trending topics
                st.subheader("Trending Topics")
                trending = extract_trending_topics(display_articles, n=10)

                if trending:
                    # Create a bar chart of trending topics
                    topic_df = pd.DataFrame(
                        {
                            "Topic": trending,
                            "Count": range(
                                len(trending), 0, -1
                            ),  # Dummy counts in descending order
                        }
                    )

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(
                        topic_df["Topic"], topic_df["Count"], color="skyblue"
                    )
                    ax.set_title("Trending Topics")
                    ax.set_xlabel("Relative Frequency")

                    st.pyplot(fig)
            else:
                st.info("No sentiment data available for analysis.")

    # Chat section
    st.header("Chat with the News")
    st.write("Ask questions about the news articles displayed above.")

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")

    # User input
    user_query = st.text_input("Your question:")

    if st.button("Ask") and user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate response based on the articles
        if st.session_state.articles:
            response = answer_question(user_query, st.session_state.articles)
        else:
            response = "Please refresh the news first to load articles."

        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Rerun to update the UI
        st.experimental_rerun()


if __name__ == "__main__":
    main()

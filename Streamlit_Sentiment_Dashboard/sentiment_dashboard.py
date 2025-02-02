#pip install streamlit transformers spacy pandas
#python -m spacy download en_core_web_sm


import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy

# Load spaCy model for key phrase extraction.
# Ensure you have downloaded it with:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis pipeline from Hugging Face.
# This uses a model fine-tuned for sentiment analysis (e.g., distilbert-base-uncased-finetuned-sst-2-english).
sentiment_pipeline = pipeline("sentiment-analysis")

# App Title and Description
st.title("Sentiment Analysis Dashboard")
st.write("Enter text (e.g., tweets, reviews) below to analyze sentiment and extract key phrases.")

# Text input area
user_text = st.text_area("Enter your text here:", height=150)

# When the user clicks the "Analyze" button, perform the analysis.
if st.button("Analyze"):
    if user_text.strip():
        # 1. Perform sentiment analysis
        result = sentiment_pipeline(user_text)[0]
        sentiment_label = result["label"]
        sentiment_score = result["score"]
        
        st.subheader("Sentiment Analysis Result")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Score:** {sentiment_score:.2f}")

        # 2. Visualize sentiment scores with a bar chart.
        # For binary sentiment, we assume the opposite probability is (1 - score).
        if sentiment_label.upper() == "POSITIVE":
            scores = {"Positive": sentiment_score, "Negative": 1 - sentiment_score}
        else:
            scores = {"Positive": 1 - sentiment_score, "Negative": sentiment_score}

        score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["Probability"])
        st.bar_chart(score_df)

        # 3. Extract key phrases using spaCy's noun chunks.
        doc = nlp(user_text)
        key_phrases = list({chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()})
        
        st.subheader("Key Phrases")
        if key_phrases:
            st.write(key_phrases)
        else:
            st.write("No key phrases found.")
    else:
        st.warning("Please enter some text for analysis.")



"""
Paragraph 1: Positive Sentiment
"I'm so excited to share my latest project with you all! It's been a labor of love, and I'm confident that it's going to make a real difference in people's lives. The feedback I've received so far has been incredibly positive, and I'm thrilled to see how it's resonating with others."
Paragraph 2: Negative Sentiment
"I'm extremely disappointed with the service I received from this company. The product was faulty, and the customer support team was unhelpful and unresponsive. I would not recommend this company to anyone, and I'm going to make sure to share my negative experience with friends and family."
Paragraph 3: Neutral Sentiment
"I recently attended a conference on machine learning, and while it was interesting to learn about the latest developments in the field, I didn't feel particularly inspired or motivated. The speakers were knowledgeable, but the content was mostly a review of existing research, and I didn't see any new insights or breakthroughs."
Paragraph 4: Mixed Sentiment
"I just got back from a trip to Europe, and while it was amazing to see the sights and experience the culture, there were some frustrating moments along the way. The language barrier was a challenge, and I had to deal with some annoying tourists who were being loud and obnoxious. However, the food was incredible, and I met some lovely people who made the trip truly special."

"""
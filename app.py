import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns




st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("Sentiment Analysis Web App")
st.subheader("This app helps you analyze the emotion behind text, like product reviews or comments. It tells you if a sentence is positive, negative, or neutral in tone.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'Text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("📄 Data Preview")
    st.write(df.head())

    if 'Text' not in df.columns:
        st.error("❌ The uploaded CSV must have a 'Text' column.")
    else:
        # Sentiment analysis function
        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity

        # Apply analysis
        df['SentimentScore'] = df['Text'].apply(get_sentiment)
        df['SentimentLabel'] = df['SentimentScore'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
        )

        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='SentimentLabel', palette='viridis', ax=ax)
        st.pyplot(fig)

        st.subheader("📁 Full Data with Sentiment")
        st.dataframe(df)

# Real-time custom input
st.subheader("✍️ Test Your Own Text")
user_text = st.text_area("Enter a sentence for sentiment analysis:")

if st.button("Analyze"):
    if user_text.strip() != "":
        result = TextBlob(user_text).sentiment
        polarity = result.polarity
        subjectivity = result.subjectivity

        # Classify with emoji and color
        if polarity > 0:
            sentiment_label = "😊 Positive"
            color = "green"
        elif polarity < 0:
            sentiment_label = "😠 Negative"
            color = "red"
        else:
            sentiment_label = "😐 Neutral"
            color = "gray"

        # Display results
        st.markdown(f"<h4 style='color:{color}'>Sentiment: {sentiment_label}</h4>", unsafe_allow_html=True)
        st.markdown(f"<b>Polarity Score:</b> `{polarity}`")
        st.markdown(f"<b>Subjectivity Score:</b> `{subjectivity}`")
    else:
        st.warning("⚠️ Please enter some text before clicking Analyze.")

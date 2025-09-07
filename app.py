import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the trained model and vectorizer
try:
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please make sure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()


# Define the emotion mapping based on your notebook
emotion_numbers = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'surprise', 4: 'fear', 5: 'joy'}


# Preprocessing functions (copied from your notebook)
def remove_punc(txt):
    return txt.translate(str.maketrans('','',string.punctuation))

def remove_numbers(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new

def remove_emojis(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

stop_words = set(stopwords.words('english'))

def remove_stopwords(txt):
    words = txt.split()
    cleaned = []
    for i in words:
        if not i in stop_words:
            cleaned.append(i)
    return ' '.join(cleaned)

def preprocess_text(text):
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text

# Streamlit App
st.title("Emotion Detection App")

st.write("Enter a sentence to detect the emotion.")

user_input = st.text_area("Enter text here:")


# Define the emotion mapping with emojis
mapping_emotion = {
    "joy": "Joy 😊",
    "sadness": "Sadness 😢",
    "anger": "Anger 😠",
    "fear": "Fear 😨",
    "love": "Love ❤️",
    "surprise": "Surprise 😲"
}

# Update emotion_numbers to use emoji labels
emotion_numbers = {
    0: mapping_emotion["sadness"],
    1: mapping_emotion["anger"],
    2: mapping_emotion["love"],
    3: mapping_emotion["surprise"],
    4: mapping_emotion["fear"],
    5: mapping_emotion["joy"]
}


if st.button("Predict Emotion"):
    if user_input:
        # Preprocess the input text
        cleaned_input = preprocess_text(user_input)

        # Vectorize the preprocessed text
        input_vector = vectorizer.transform([cleaned_input])

        # Predict the emotion
        prediction = model.predict(input_vector)[0]

        # Get the emotion label
        predicted_emotion = emotion_numbers.get(prediction, "Unknown")

        st.success(f"Predicted Emotion: {predicted_emotion}")

        # Add special effects based on emotion
        if predicted_emotion == "Joy 😊":
            st.balloons()
        elif predicted_emotion == "Sadness 😢":
            st.snow()
        elif predicted_emotion == "Anger 😠":
            st.error("Anger detected!")
        elif predicted_emotion == "Fear 😨":
            st.warning("Fear detected!")
        elif predicted_emotion == "Love ❤️":
            st.write("Spread the love!")
        elif predicted_emotion == "Surprise 😲":
            st.info("Surprise detected!")
        else:
            st.write("No special effects for this emotion.")
    else:
        st.warning("Please enter some text to predict the emotion.")
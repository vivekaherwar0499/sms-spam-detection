import nltk
nltk.data.path.append("nltk_data") 
nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')

import streamlit as st
import pickle
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = nltk.word_tokenize(text)
    words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Streamlit App
st.title("📱 SMS Spam Detection App")

user_input = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # Preprocess input
        processed = preprocess_text(user_input)
        # Convert to vector
        vectorized = vectorizer.transform([processed])
        # Predict
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")
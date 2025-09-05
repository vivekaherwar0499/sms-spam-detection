import nltk
import os
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- NLTK Setup for Streamlit Cloud ----------------
# Create a folder in current working directory to store nltk data
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

# Tell nltk to look here
nltk.data.path.append(nltk_data_path)

# Download required NLTK data into this folder
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)   
nltk.download('stopwords', download_dir=nltk_data_path)
# -----------------------------------------------------------------

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

# ---------------- Streamlit App ----------------
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
import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("sms-spam.csv", encoding="latin-1")

# Rename correct columns
df = df.rename(columns={"ï»¿v1": "label", "v2": "message"})

# Keep only useful columns
df = df[["label", "message"]]

# Convert labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing function
ps = PorterStemmer()
def preprocess_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = nltk.word_tokenize(text)
    words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Features and labels
X = df['message']
y = df['label']

# Vectorize
tfidf = TfidfVectorizer()
X_vectorized = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and vectorizer saved successfully!")
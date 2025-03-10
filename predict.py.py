import pickle
from src.data_preprocessing import clean_text
from sklearn.feature_extraction.text import CountVectorizer

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_sentiment(text):
    """Predict sentiment for a given text."""
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    return "Positive" if prediction == 0 else "Negative"

if __name__ == "__main__":
    text = input("Enter a tweet: ")
    print(predict_sentiment(text))

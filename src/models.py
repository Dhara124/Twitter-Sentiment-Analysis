import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_model():
    # Load dataset
    tweets_df = pd.read_csv("data/twitter.csv")
    tweets_df['clean_tweet'] = tweets_df['tweet'].apply(clean_text)

    # Convert text to numerical data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets_df['clean_tweet'])
    y = tweets_df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, vectorizer

if __name__ == "__main__":
    train_model()
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(message):
    """Remove punctuation and stopwords from a message."""
    message = ''.join([char for char in message if char not in string.punctuation])
    return ' '.join([word.lower() for word in message.split() if word.lower() not in stopwords.words('english')])

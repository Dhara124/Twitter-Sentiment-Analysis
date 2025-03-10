import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the input text by:
    - Removing punctuation
    - Removing stopwords
    - Converting to lowercase
    """
    text_no_punct = ''.join([char for char in text if char not in string.punctuation])
    words = text_no_punct.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in stopwords.words('english')]
    return ' '.join(cleaned_words)

def load_dataset(file_path):
    """
    Loads the dataset from a CSV file and returns a Pandas DataFrame.
    """
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix using Seaborn.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def tokenize_and_vectorize(texts):
    """
    Converts a list of texts into a matrix of token counts using CountVectorizer.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_test_split_data(X, y, test_size=0.2):
    """
    Splits data into train and test sets.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=42)


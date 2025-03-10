import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(message):
    """Remove punctuation and stopwords from a message."""
    message = ''.join([char for char in message if char not in string.punctuation])
    return ' '.join([word.lower() for word in message.split() if word.lower() not in stopwords.words('english')])

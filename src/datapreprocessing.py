import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Setup NLTK
nltk.download('stopwords')
stemmer = SnowballStemmer('english')
stops = set(stopwords.words('english'))

# Retain negations in stopwords
not_stop = [
    "aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't",
    "haven't", "isn't", "mightn't", "mustn't", "needn't", "no", "nor", "not",
    "shan't", "shouldn't", "wasn't", "weren't", "wouldn't"
]
for word in not_stop:
    if word in stops:
        stops.remove(word)

def review_to_words(raw_review):
    """Convert a raw review to a clean review."""
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stops]
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return ' '.join(stemming_words)

def clean_review_column(data):
    print("Cleaning review text...")
    data = data.dropna(axis=0)
    data = data[~data['condition'].str.contains('</span>', na=False)].reset_index(drop=True)
    data['review_clean'] = data['review'].apply(review_to_words)
    return data

def add_features(data):
    print("Adding engineered features...")

    # Time features (assume 'date' already in datetime format)
    data['Year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Sentiment labels
    data['Review_Sentiment'] = (data['rating'] >= 5).astype(int)

    # Text-based features
    data['count_sent'] = data["review"].apply(lambda x: len(re.findall("\n", str(x))) + 1)
    data['count_word'] = data["review_clean"].apply(lambda x: len(str(x).split()))
    data['count_unique_word'] = data["review_clean"].apply(lambda x: len(set(str(x).split())))
    data['count_letters'] = data["review_clean"].apply(lambda x: len(str(x)))
    data["count_punctuations"] = data["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    data["count_words_upper"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    data["count_words_title"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    data["count_stopwords"] = data["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))
    data["mean_word_len"] = data["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # Season feature: 1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter
    data['season'] = data["month"].apply(lambda x: 1 if 3 <= x <= 5 else (2 if 6 <= x <= 8 else (3 if 9 <= x <= 11 else 4)))

    return data

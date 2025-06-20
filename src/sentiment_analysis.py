import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import CountVectorizer

def compute_textblob_sentiment(data):
    print("Running TextBlob Sentiment Analysis...")

    reviews = data['review_clean']
    Predict_Sentiment = []

    for review in tqdm(reviews, desc="Analyzing sentiment polarity"):
        blob = TextBlob(review)
        Predict_Sentiment.append(blob.sentiment.polarity)

    data["Predict_Sentiment"] = Predict_Sentiment
    data["Predict_Sentiment2"] = Predict_Sentiment.copy()  # redundant but replicating original logic

    print("\n--- Correlations ---")
    print("Correlation: Predict_Sentiment vs Rating")
    print(np.corrcoef(data["Predict_Sentiment"], data["rating"]))
    print("Correlation: Predict_Sentiment vs Review_Sentiment")
    print(np.corrcoef(data["Predict_Sentiment"], data["Review_Sentiment"]))

    return data


def compute_dictionary_sentiment(df_test):
    print("Running Dictionary-Based Sentiment Analysis...")

    # Load dictionary
    word_table = pd.read_csv("D:/Shivani/Database/dictionary/inquirerbasic.csv")

    # Extract positive and negative word lists
    temp_Positiv = []
    for i in range(len(word_table.Positiv)):
        if word_table.iloc[i, 2] == "Positiv":
            word = word_table.iloc[i, 0].lower()
            word = re.sub(r'\d+', '', word)
            word = re.sub('#', '', word)
            temp_Positiv.append(word)
    Positiv_word_list = list(set(temp_Positiv))

    temp_Negativ = []
    for i in range(len(word_table.Negativ)):
        if word_table.iloc[i, 3] == "Negativ":
            word = word_table.iloc[i, 0].lower()
            word = re.sub(r'\d+', '', word)
            word = re.sub('#', '', word)
            temp_Negativ.append(word)
    Negativ_word_list = list(set(temp_Negativ))

    # Count positive words
    vectorizer = CountVectorizer(vocabulary=Positiv_word_list)
    X = vectorizer.fit_transform(df_test['review_clean'])
    df_test["num_Positiv_word"] = pd.DataFrame(X.toarray()).sum(axis=1)

    # Count negative words
    vectorizer2 = CountVectorizer(vocabulary=Negativ_word_list)
    X2 = vectorizer2.fit_transform(df_test['review_clean'])
    df_test["num_Negativ_word"] = pd.DataFrame(X2.toarray()).sum(axis=1)

    # Compute sentiment ratio
    df_test["Positiv_ratio"] = df_test["num_Positiv_word"] / (
        df_test["num_Positiv_word"] + df_test["num_Negativ_word"] + 1e-6
    )
    df_test["sentiment_by_dic"] = df_test["Positiv_ratio"].apply(lambda x: 1 if x >= 0.5 else 0)

    return df_test

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation

from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def run_deep_model(data):
    print("Running Deep Learning Model...")

    # Split data
    df_train, df_test = train_test_split(data, test_size=0.25, random_state=0)
    y_train = df_train['Review_Sentiment']
    y_test = df_test['Review_Sentiment']

    # Bag of Words vectorization (4-grams)
    cv = CountVectorizer(max_features=20000, ngram_range=(4, 4))
    pipeline = Pipeline([('vect', cv)])
    df_train_features = pipeline.fit_transform(df_train['review_clean'])
    df_test_features = pipeline.transform(df_test['review_clean'])

    print("Train feature shape:", df_train_features.shape)
    print("Test feature shape:", df_test_features.shape)

    # Deep model
    model = Sequential()
    model.add(Dense(200, input_shape=(20000,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    # Train
    hist = model.fit(df_train_features, y_train, epochs=10, batch_size=128)

    # Plot training
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()

    # Predict
    sub_preds_deep = model.predict(df_test_features, batch_size=32).flatten()

    # Save model
    pickle.dump(model, open('model.pkl', 'wb'))

    return data, df_train, df_test, sub_preds_deep


def run_lgbm_model(df_train, df_test):
    print("Running LightGBM Model...")

    target = df_train['Review_Sentiment']
    feats = [
        'usefulCount', 'day', 'Year', 'month',
        'Predict_Sentiment', 'Predict_Sentiment2',
        'count_sent', 'count_word', 'count_unique_word', 'count_letters',
        'count_punctuations', 'count_words_upper', 'count_words_title',
        'count_stopwords', 'mean_word_len', 'season'
    ]

    sub_preds = np.zeros(df_test.shape[0])
    trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42)

    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.10,
        num_leaves=30,
        subsample=0.9,
        max_depth=7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        min_child_weight=2,
        verbose=-1
    )

    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        verbose=100,
        early_stopping_rounds=100
    )

    sub_preds = clf.predict(df_test[feats])
    print(confusion_matrix(y_pred=sub_preds, y_true=df_test['Review_Sentiment']))
    print(classification_report(y_pred=sub_preds, y_true=df_test['Review_Sentiment']))

    # Feature importance
    feature_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cols = feature_importance_df.groupby("feature").mean().sort_values(by="importance", ascending=False).index
    best_features = feature_importance_df[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    plt.show()

    return sub_preds

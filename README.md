# Sentimental Analysis of Drug Reviews

A machine learning-based web application that recommends the most suitable drugs based on user-input health conditions, using sentiment analysis of patient reviews.

## Project Overview

This project leverages Natural Language Processing (NLP), Recurrent Neural Networks (RNN), Bi-directional LSTM, and the LightGBM classifier to build a hybrid recommendation system. Given a health condition, it analyzes real-world drug reviews to recommend medications with the highest predicted positive sentiment and effectiveness.

Developed as a final year capstone project for the B.E. in Information Technology program at Shah & Anchor Kutchhi Engineering College.

## Objectives

- Clean and preprocess real-world drug review data
- Perform exploratory data analysis (EDA)
- Build sentiment analysis models using RNN and Bi-LSTM
- Classify sentiments using LightGBM
- Develop a Flask-based frontend for user interaction
- Recommend top-rated drugs based on hybrid prediction

## Tech Stack

- **Languages**: Python, HTML/CSS, JavaScript
- **Libraries**: 
  - Data Processing: `pandas`, `numpy`, `BeautifulSoup`, `re`
  - NLP: `NLTK`, `TextBlob`, `sklearn`
  - Deep Learning: `Keras`, `TensorFlow`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - ML Models: `LightGBM`, `sklearn`
- **Web Framework**: Flask
- **Database**: MySQL


## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- **Name**: Drug Review Dataset (Drugs.com)
- **Size**: 215,063 reviews
- **Fields**: `drugName`, `condition`, `review`, `rating`, `date`, `usefulCount`

## Features

- Sentiment classification using TextBlob and Harvard Emotion Lexicon
- RNN-BiLSTM deep learning for review polarity prediction
- LightGBM classifier for efficient binary classification
- Real-time drug search interface using Flask & MySQL
- Integrated web scraping and condition-specific recommendation logic

## Screenshots

Find all EDA visualizations and app output screenshots in the screenshots folder.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/pill-recommendation-system.git
cd pill-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Start the Flask app
python app.py






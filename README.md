# 💊 Sentimental Analysis of Drug Reviews

This project performs sentiment analysis on patient drug reviews collected from [Drugs.com](https://www.drugs.com/). It uses natural language processing, deep learning, machine learning (LightGBM), and dictionary-based sentiment to identify and aggregate positive and negative feedback for each drug-condition combination.

---

## 🧠 Project Objectives

- Extract and clean user reviews
- Generate visual insights (WordClouds, ratings, sentiment)
- Train a deep learning model to classify sentiment
- Use LightGBM for numeric feature-based prediction
- Combine predictions from multiple sources
- Rank drugs per condition using aggregated sentiment

---

## 🗂️ Folder Structure

Sentimental-Analysis-of-Drug-Reviews/
│
├── app/ # (web UI)
├── data/ # Raw data 
│ └── inquirerbasic.csv
├── screenshots/ # Visualizations and result screenshots
├── src/ # Source code
│ ├── main.py
│ ├── load_data.py
│ ├── eda.py
│ ├── text_preprocessing.py
│ ├── model_training.py
│ ├── sentiment_analysis.py
│ └── results.py
├── requirements.txt # Python dependencies
└── README.md # You're here

---


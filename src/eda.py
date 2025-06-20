import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

def explore_data(data):
    print("\n--- Dataset Info ---")
    print(data.info())
    print("\n--- Null Values ---")
    print(data.isnull().sum())
    print("\n--- Descriptive Stats ---")
    print(data.describe())
    print("\n--- Sample Rows ---")
    print(data.sample(5))


def generate_wordclouds(data):
    stopwords = set(STOPWORDS)

    # WordCloud: Most Common Conditions
    wordcloud = WordCloud(background_color='lightblue', stopwords=stopwords, max_words=100, width=1200, height=800)
    wordcloud.generate(str(data['condition']))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud)
    plt.title('Most Common Conditions among the Patients', fontsize=30)
    plt.axis('off')
    plt.show()

    # WordCloud: Most Popular Drugs
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, width=1200, height=800)
    wordcloud.generate(str(data['drugName']))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud)
    plt.title('Most Popular Drugs', fontsize=30)
    plt.axis('off')
    plt.show()

    # WordCloud: Most Common Words in Reviews
    wordcloud = WordCloud(background_color='yellow', stopwords=stopwords, width=1200, height=800)
    wordcloud.generate(str(data['review']))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud)
    plt.title('Most Common Words in Reviews', fontsize=30)
    plt.axis('off')
    plt.show()


def generate_all_plots(data):
    # Most drugs per condition
    data.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False).head(40).plot.bar(figsize=(19, 7), color='crimson')
    plt.title('Most Drugs per Condition', fontsize=30)
    plt.xlabel('Condition')
    plt.ylabel('Unique Drug Count')
    plt.show()

    # Drugs used for many conditions
    data.groupby(['drugName'])['condition'].nunique().sort_values(ascending=False).head(40).plot.bar(figsize=(19, 7), color='violet')
    plt.title('Drugs Used for Many Conditions', fontsize=30)
    plt.xlabel('Drug Name')
    plt.ylabel('Condition Count')
    plt.show()

    # Condition Frequency
    data['condition'].value_counts().head(40).plot.bar(figsize=(19, 7), color='purple')
    plt.title('Most Common Conditions', fontsize=30)
    plt.xlabel('Condition')
    plt.ylabel('Count')
    plt.show()

    # Ratings Donut Chart
    size = [68005, 46901, 36708, 25046, 12547, 10723, 8462, 6671]
    colors = ['pink', 'cyan', 'maroon', 'magenta', 'orange', 'lightblue', 'lightgreen', 'yellow']
    labels = ["10", "1", "9", "8", "7", "5", "6", "4"]
    circle = plt.Circle((0, 0), 0.7, color='white')
    plt.figure(figsize=(10, 10))
    plt.pie(size, colors=colors, labels=labels, autopct='%.2f%%')
    plt.gca().add_artist(circle)
    plt.axis('off')
    plt.title('Ratings Distribution', fontsize=30)
    plt.legend()
    plt.show()

    # Review Sentiment Pie Chart
    data['Review_Sentiment'] = (data['rating'] >= 5).astype(int)
    size = data['Review_Sentiment'].value_counts().tolist()
    colors = ['yellow', 'skyblue']
    labels = ["Positive Sentiment", "Negative Sentiment"]
    explode = [0, 0.1]
    plt.figure(figsize=(10, 10))
    plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%')
    plt.axis('off')
    plt.title('Sentiment Distribution', fontsize=30)
    plt.legend()
    plt.show()

    # WordClouds for Sentiments
    for sentiment_value, title, bg_color in [(1, 'Positive Reviews', 'lightgreen'), (0, 'Negative Reviews', 'grey')]:
        text = " ".join(data['review'][data['Review_Sentiment'] == sentiment_value])
        wordcloud = WordCloud(background_color=bg_color, stopwords=STOPWORDS, width=1200, height=800).generate(text)
        plt.figure(figsize=(15, 15))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title(f'Most Common Words in {title}', fontsize=30)
        plt.show()

    # Time-based features
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['Year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Yearly review counts
    plt.figure(figsize=(19, 8))
    sns.countplot(x=data['Year'], palette='dark')
    plt.title('Reviews Per Year', fontsize=30)
    plt.xlabel('Year')
    plt.ylabel('Review Count')
    plt.show()

    # Yearly rating distribution
    plt.figure(figsize=(19, 8))
    sns.boxplot(x=data['Year'], y=data['rating'], palette='dark')
    plt.title('Rating Distribution per Year', fontsize=30)
    plt.xlabel('Year')
    plt.ylabel('Ratings')
    plt.show()

    # Yearly sentiment distribution
    plt.figure(figsize=(19, 8))
    sns.violinplot(x=data['Year'], y=data['Review_Sentiment'])
    plt.title('Sentiment Distribution per Year', fontsize=30)
    plt.xlabel('Year')
    plt.ylabel('Sentiments')
    plt.show()

    # Monthly review counts
    plt.figure(figsize=(19, 8))
    sns.countplot(x=data['month'], palette='pastel')
    plt.title('Reviews Per Month', fontsize=30)
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.show()

    # Monthly rating distribution
    plt.figure(figsize=(19, 8))
    sns.boxplot(x=data['month'], y=data['rating'], palette='pastel')
    plt.title('Rating Distribution per Month', fontsize=30)
    plt.xlabel('Month')
    plt.ylabel('Ratings')
    plt.show()

    # Monthly sentiment distribution
    plt.figure(figsize=(19, 8))
    sns.violinplot(x=data['month'], y=data['rating'], palette='pastel')
    plt.title('Sentiment Distribution per Month', fontsize=30)
    plt.xlabel('Month')
    plt.ylabel('Sentiments')
    plt.show()

    # Daily review counts
    plt.figure(figsize=(19, 8))
    sns.countplot(x=data['day'], palette='colorblind')
    plt.title('Reviews Per Day', fontsize=30)
    plt.xlabel('Day')
    plt.ylabel('Review Count')
    plt.show()

    # Daily rating distribution
    plt.figure(figsize=(19, 8))
    sns.boxplot(x=data['day'], y=data['rating'], palette='colorblind')
    plt.title('Rating Distribution per Day', fontsize=30)
    plt.xlabel('Day')
    plt.ylabel('Ratings')
    plt.show()

    # Daily sentiment distribution
    plt.figure(figsize=(19, 8))
    sns.violinplot(x=data['day'], y=data['rating'])
    plt.title('Sentiment Distribution per Day', fontsize=30)
    plt.xlabel('Day')
    plt.ylabel('Sentiments')
    plt.show()

    # Distribution of Useful Counts
    plt.figure(figsize=(15, 8))
    sns.histplot(data['usefulCount'], kde=False, bins=100, color='red')
    plt.title('Distribution of Useful Counts', fontsize=30)
    plt.xlabel('Useful Count')
    plt.ylabel('Frequency')
    plt.show()

    # Stacked bar: sentiment per year
    df_year = pd.crosstab(data['Year'], data['Review_Sentiment'])
    df_year.div(df_year.sum(1), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8), color=['red', 'orange'])
    plt.title('Sentiment Distribution by Year', fontsize=30)
    plt.xlabel('Year')
    plt.show()

    # Stacked bar: sentiment per month
    df_month = pd.crosstab(data['month'], data['Review_Sentiment'])
    df_month.div(df_month.sum(1), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8), color=['darkblue', 'violet'])
    plt.title('Sentiment Distribution by Month', fontsize=30)
    plt.xlabel('Month')
    plt.show()

    # Stacked bar: sentiment per day
    df_day = pd.crosstab(data['day'], data['Review_Sentiment'])
    df_day.div(df_day.sum(1), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8), color=['lightblue', 'yellow'])
    plt.title('Sentiment Distribution by Day', fontsize=30)
    plt.xlabel('Day')
    plt.show()

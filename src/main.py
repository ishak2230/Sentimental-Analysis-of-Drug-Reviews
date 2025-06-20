from data.load_data import load_and_combine_data
from eda.basic_eda import explore_data
from eda.wordclouds import generate_wordclouds
from eda.plots import generate_all_plots
from preprocessing.clean_reviews import clean_review_column
from preprocessing.feature_engineering import add_features
from modeling.deep_model import run_deep_model
from modeling.lgbm_model import run_lgbm_model
from sentiment.textblob_sentiment import compute_textblob_sentiment
from sentiment.dictionary_sentiment import compute_dictionary_sentiment
from utils.save_results import save_final_output

def main():
    # Step 1: Load and combine data
    data, train, test = load_and_combine_data()

    # Step 2: Basic EDA
    explore_data(data)

    # Step 3: Generate WordClouds
    generate_wordclouds(data)

    # Step 4: Visualization Plots
    generate_all_plots(data)

    # Step 5: Clean review text
    data = clean_review_column(data)

    # Step 6: Add engineered features
    data = add_features(data)

    # Step 7: Run Deep Learning Model
    data, df_train, df_test, deep_preds = run_deep_model(data)

    # Step 8: Run LightGBM Model
    lgb_preds = run_lgbm_model(df_train, df_test)

    # Step 9: TextBlob Sentiment
    data = compute_textblob_sentiment(data)

    # Step 10: Dictionary-based Sentiment
    df_test = compute_dictionary_sentiment(df_test)

    # Step 11: Save final output
    save_final_output(df_test, deep_preds, lgb_preds)

if __name__ == "__main__":
    main()

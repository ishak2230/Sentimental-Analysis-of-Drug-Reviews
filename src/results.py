import pandas as pd

def save_final_output(df_test, deep_preds, lgb_preds):
    print("Saving final results...")

    # If not already normalized
    if 'usefulCount' not in df_test.columns or 'user_size' not in df_test.columns:
        df_test = add_user_count(df_test)

    # Ensure required columns exist
    if 'sentiment_by_dic' not in df_test.columns:
        raise ValueError("Missing sentiment_by_dic. Ensure dictionary-based sentiment is computed.")

    df_test['deep_pred'] = deep_preds
    df_test['machine_pred'] = lgb_preds

    # Combine predictions
    df_test['total_pred'] = (df_test['deep_pred'] + df_test['machine_pred'] + df_test['sentiment_by_dic']) * df_test['usefulCount']

    # Aggregate by condition + drugName
    final_df = df_test.groupby(['condition', 'drugName']).agg({'total_pred': 'mean'})
    pd.set_option('display.max_rows', 5000)
    print(final_df)

    # Save output
    output_path = 'D:/Shivani/Database/output.csv'
    final_df.to_csv(output_path)
    print(f"Saved output to {output_path}")


def add_user_count(data):
    grouped = data.groupby(['condition']).size().reset_index(name='user_size')
    data = pd.merge(data, grouped, on='condition', how='left')
    data['usefulCount'] = data['usefulCount'] / data['user_size']
    return data

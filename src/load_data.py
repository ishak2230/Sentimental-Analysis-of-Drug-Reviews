import pandas as pd

def load_and_combine_data():
    # Load training and test data
    train = pd.read_csv('D:/Shivani/Database/drugsComTrain_raw.csv')
    test = pd.read_csv('D:/Shivani/Database/drugsComTest_raw.csv')

    # Print shapes
    print("Shape of train:", train.shape)
    print("Shape of test:", test.shape)

    # Combine train and test for unified analysis
    data = pd.concat([train, test])
    print("Combined data shape:", data.shape)

    return data, train, test

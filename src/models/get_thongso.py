import pandas as pd
import os

def main():
    data_path = '/home/ponydasierra/thuctap/news_classification/data/processed5'
    # train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    # val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
    # test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    df = pd.read_csv(os.path.join(data_path, 'clean_data_final.csv'))
    # print(f"Train data count: {len(train_df)}")
    # print(f"Validation data count: {len(val_df)}")
    # print(f"Test data count: {len(test_df)}")
    # print(f"Avg. text length in train data: {train_df['full_text'].str.len().mean():.2f}")
    # print(f"Top 3 users in train data: {train_df['user'].value_counts().head(3)}")
    # print(f"Top 3 domain in train data: {train_df['domain'].value_counts().head(3)}")

    # #Drop unnecessary columns
    # df = df.drop(columns=['fake_ratio [%]', 'domain', 'title', 'source', 'hour', 'day_of_week', 'cleaned_title', 'cleaned_content'])
    # # Rename columns tokenizer_title and tokenizer_content to title and content
    # df = df.rename(columns={'tokenized_title': 'title', 'tokenized_content': 'content'})
    # # df = df.rename(columns={'title': 'cleaned_title', 'content': 'cleaned_content'})


    print(f"Dataset Info:")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns}")
    #save the cleaned data
    df.to_csv(os.path.join(data_path, 'clean_data_final.csv'), index=False)
    # print(f"Total samples: {len(df)}")
    # print("Label distribution:")
    # print(df['label'].value_counts())

if __name__ == "__main__":
    main()

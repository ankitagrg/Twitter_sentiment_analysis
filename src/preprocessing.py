import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

def clean_text(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    stop_words = set(stopwords.words('english'))
    
    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words)
    )
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_data("../data/raw_tweets.csv", "../data/cleaned_tweets.csv")

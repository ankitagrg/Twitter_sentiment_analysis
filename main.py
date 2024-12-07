from src.preprocessing import preprocess_data
from src.train import train_model

if __name__ == "__main__":
    # Step 1: Preprocess data
    print("Preprocessing data...")
    preprocess_data("data/raw_tweets.csv", "data/cleaned_tweets.csv")
    
    # Step 2: Train and evaluate model
    print("Training model...")
    train_model("data/cleaned_tweets.csv")
    print("Sentiment analysis complete!")

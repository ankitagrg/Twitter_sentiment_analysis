import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

from utils import load_data

def train_model(data_path):
    # Load preprocessed data
    df = load_data(data_path)
    X = df['cleaned_text']
    y = df['sentiment']  # Assume 'sentiment' column contains labels
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save model and vectorizer
    joblib.dump(model, '../models/sentiment_model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')
    
    # Save report
    with open("../results/classification_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    train_model("../data/cleaned_tweets.csv")

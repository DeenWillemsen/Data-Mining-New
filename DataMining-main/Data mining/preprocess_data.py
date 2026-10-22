import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

def preprocess_data():

    #load dataset
    df = pd.read_csv("DataMining-main\\Data mining\\all_data.csv")

    #filter on english
    df = df[df["Review_Language"] == "English"].copy()

    #target variables: 0 -> real, 1 -> fake
    y = df["source"]

    #data of review has to be altered (review existed of two columns)
    df["review_text"] = (
        df["Upside_Review"].fillna("") + " " +
        df["Downside_Review"].fillna("")
    )

    X = df["review_text"]
    sentiments = df["Sentiment"]

    #training/test split
    X_train, X_test, y_train, y_test, sent_train, sent_test = train_test_split(
        X,
        y,
        sentiments,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    #lemmatize
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        words = re.findall(r"\b\w\w+\b", text.lower())
        words = [w if w in ENGLISH_STOP_WORDS else lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    X_train_processed = X_train.apply(lemmatize_text)
    X_test_processed = X_test.apply(lemmatize_text)

    #Bag of words
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2
    )

    X_train_bow = vectorizer.fit_transform(X_train_processed)
    X_test_bow = vectorizer.transform(X_test_processed)


    return X_train_bow, X_test_bow, y_train, y_test, vectorizer, sent_test

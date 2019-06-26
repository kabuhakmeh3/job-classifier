# Tools to facilitate text processing
# Several preprocessing steps are here
# Use these as part of the larger workflow

import os
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def get_df_tokens(df, token_col):
    '''Returns a column of tokens from a column containing sentences

    Call should assign output to a new column
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    return df[token_col].apply(tokenizer.tokenize)

def combine_text_columns(col_names):
    '''Combine the tokenized words from multiple columns into one (test first)
    '''
    return "this function is in development"


def create_count_vectorizer(data):
    '''Create a count vectorizer model

    Returns:
    + Document-term matrix [X]
    + CountVectorizer

    This is for model training only

    Use a pre-fit count vectorizer in production
    '''
    count_vectorizer = CountVectorizer()

    X = count_vectorizer.fit_transform(data)

    return X, count_vectorizer

def create_tfidf_vectorizer(data):
    '''Create a Tf-idf vectorizer model

    Returns:
    + Tf-idf-weighted document-term matrix [X]
    + CountVectorizer

    This is for model training only

    Use a pre-fit count vectorizer in production
    '''
    tfidf_vectorizer = TfidfVectorizer()

    X = tfidf_vectorizer.fit_transform(data)

    return X, tfidf_vectorizer

def get_cv_test_counts(X_test, cv_model):
    '''Apply existing CV model to data

    Returns:
    + Transformed X

    Transform data in production

    Works for both CountVectorizer and TfidfVectorizer

    No need to specify which one it is

    Note: Update function name from cv to vectorizer
    '''
    X_test_counts = cv_model.transform(X_test)
    return X_test_counts

import os
import pandas as pd

## install these on ec2
#import keras
#import nltk
#import re
#import codecs

from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical

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

import os, sys, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

# This version trains models and saves locally
#
# Features
# + Bag of words
# + Logistic Regression
# + Multinomial Naive Bayes
# + Complement Naive Bayes
#
# To do
# + develop a full-scale testing/evaluation pipleline
# + Make aws/s3 capable
#
# Suggestions
# + add options to choose model
# + add option to choose vectorizer

def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

def main():
    '''Create and save model
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt
    import nlp_preprocessing as nlp

    # if pulling from s3
    #path_to_data = '../.keys/'
    #file_name = 'csv_to_classify.json'
    #bucket, key = bt.load_s3_location(path_to_data, file_name)
    #df = bt.load_df_from_s3(bucket, key, compression='gzip')

    # local data
    path_to_data = '../data/multiclass/'
    data_file = 'labeled_large.csv'
    df = pd.read_csv(os.path.join(path_to_data, data_file))
    print('loaded CSV')
    # remove samples not of interestc

    # map roles to general labels
    key_path = '../.keys'
    role_mapper = load_pickle(os.path.join(key_path, 'roles.pickle'))

    role_names = [n for n in role_mapper]
    df = df[df['role'].isin(role_names)]

    df['label'] = df['role'].map(role_mapper)
    df = df.dropna()

    cols_to_train = ['title', 'label']
    df = df[cols_to_train]

    # standardize text format
    df = nlp.standardize_text(df, 'title')

    # select data to predict from
    X = df['title'].tolist()
    y = df['label'].tolist()
    #y = df['gig'].tolist()

    # dividing X, y into train and test data
    print('vectorizing bag of words model')
    X_counts, count_vectorizer = nlp.create_count_vectorizer(X) # count vec
    # dos BOW need to be re-serialized?

    # training a Naive Bayes classifier
    print('testing model')
    #model = MultinomialNB().fit(X_counts, y)
    model = ComplementNB().fit(X_counts, y)

    # save model for later use (locally & on s3)
    file_to_write = '../models/complement_nb_model.pckl'
    pickle.dump(model, open(file_to_write, 'wb'))
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()

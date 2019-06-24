import os, sys, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# This version currently trains and saves a model locally
#
# Features
# + Bag of words
# + Logistic Regression
#
# To do
# + develop a full-scale testing/evaluation pipleline
# + Make aws/s3 capable
#
# Suggestions
# + add options to choose model
# + add option to choose vectorizer 

def main():
    '''Create and save model
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt
    import nlp_preprocessing as nlp

    print('Creating model from full dataset...\n')

    # if pulling from s3
    #path_to_data = '../.keys/'
    #file_name = 'csv_to_classify.json'
    #bucket, key = bt.load_s3_location(path_to_data, file_name)
    #df = bt.load_df_from_s3(bucket, key, compression='gzip')

    # local data
    path_to_data = '../data/'
    data_file = 'labeled_eda_sample_data_file.csv'
    df = pd.read_csv(os.join.path(path_to_data, data_file))

    # standardize text format
    cols_to_model = ['title']
    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    # select data to predict from
    X = df['title'].tolist()
    y = df['gig'].tolist()

    # fit CountVectorizer
    X_cv, cv = nlp.create_count_vectorizer(X)

    # fit LogisticRegression
    clf = LogisticRegression(C=30.0, class_weight='balanced',
                             solver='newton-cg', multi_class='ovr',
                             n_jobs=-1, random_state=40)
    clf.fit(X_cv, y)

    # save model for later use (locally & on s3)
    file_to_write = 'full_model.pckl'
    pickle.dump(model, open(file_to_write, 'wb'))
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()

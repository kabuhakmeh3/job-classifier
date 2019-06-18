import os, sys, time, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# add custom modules to path
path_to_module = '/home/ubuntu/job-classifier/tools/'
sys.path.append(path_to_module)
import bototools as bt
import xmltools as xt
import nlp_preprocessing as nlp


def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

def main():
    '''Run classification on job data

    Process directly from partner feeds

    Current configuration:

    + Load csv data from xml feed
    + Count Vectorize with Bag of Words
    + Predict Linear Regression
    + Select results on LinReg class probability
    + Write results back to s3
    '''

    # define paths
    key_path = '/home/ubuntu/job-classifier/.keys/'
    model_path = '/home/ubuntu/job-classifier/models/'

    # dict with partner:url pairs
    partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
    s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))

    bucket = s3_details['csv_bucket']
    file_to_write = s3_details['target']

    cv_model = load_pickle(os.path.join(model_path,'CV_lr_bow_train_only_model.pckl'))
    clf_model = load_pickle(os.path.join(model_path, 'lr_bow_train_only_model.pckl'))


    # pull xml from url and parse into df
    for partner in partners:
        url = partners[partner]
        if url.endswith('.xml.gz'):
            df = xt.xml_from_url_compressed(url)
        else:
            df = xt.xml_from_url(url)

        # standardize text format
        cols_to_model = ['title']
        for col in cols_to_model:
            df = nlp.standardize_text(df, col)

        # select data to predict from
        X_classify = df['title'].tolist()

        # get count vec
        X_classify_counts = nlp.get_cv_test_counts(X_classify, cv_model)

        # predict with model
        y_predicted = clf_model.predict(X_classify_counts)

        # assign predictions to jobs & prune dataframe
        df['gig'] = y_predicted
        cols_to_write = ['company','title','city','state','url']
        df_to_write = df[df['gig']==1][cols_to_write]

        # write jobs to accessible location on s3
        key_to_write = partner + '/' + file_to_write
        bt.write_df_to_s3(df_to_write, bucket, key_to_write, comp=False)

if __name__ == '__main__':
    main()

import os, sys, time, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main():
    '''Run classification on job data

    Current configuration:

    + Load csv data from s3
    + Count Vectorize with Bag of Words
    + Predict Linear Regression
    + Select results on LinReg class probability
    + Write results back to s3
    '''

    # add custom modules to path
    path_to_module = '/home/ubuntu/job-classifier/tools/'
    sys.path.append(path_to_module)

    # load s3 connector & preprocessing functions
    import bototools as bt
    import xmltools as xt
    import nlp_preprocessing as nlp

    # load job data from s3 csv to dataframe
    path_to_data = '/home/ubuntu/job-classifier/.keys/'
    file_name = 'cron_data_file.json'
    bucket, key, url, target = bt.load_s3_location(path_to_data, file_name)

    # pull xml from url and parse into df
    df = xt.xml_from_url(url)

    # standardize text format
    cols_to_model = ['title']
    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    # select data to predict from
    X_classify = df['title'].tolist()

    # define model path
    path_to_models = '/home/ubuntu/job-classifier/models/'

    # load count vectorizer & transform data to predict
    cv_pickle = 'CV_lr_bow_train_only_model.pckl'
    cv_path = os.path.join(path_to_models, cv_pickle)
    cv_model = pickle.load(open(cv_path, 'rb'))
    X_classify_counts = nlp.get_cv_test_counts(X_classify, cv_model)

    # load, train, fit model
    clf_pickle = 'lr_bow_train_only_model.pckl'
    clf_path = os.path.join(path_to_models, clf_pickle)
    clf_model = pickle.load(open(clf_path, 'rb'))
    y_predicted = clf_model.predict(X_classify_counts)
    y_prob = clf_model.predict_proba(X_classify_counts)

    # assign predictions to jobs & prune dataframe
    df['gig'] = y_predicted
    cols_to_write = ['company','title','city','state','url']

    df_to_write = df[df['gig']==1][cols_to_write]

    # write jobs to accessible location on s3
    # custom name by date -- test overlap between days
    timestr = time.strftime("%Y-%m-%d")
    prefix, fn = target.split('/')
    file_to_write = prefix + '/' + timestr + '-' + fn
    bt.write_df_to_s3(df_to_write, bucket, file_to_write, comp=False)

    # add labeled samples to validate for future training
    df_positive = df[df['gig']==1].sample(1000)
    file_positive = 'positive' + '/' + timestr + '-' + fn
    bt.write_df_to_s3(df_positive, bucket, file_positive, comp=False)

    df_negative = df[df['gig']==0].sample(1000)
    file_negative = 'negative' + '/' + timestr + '-' + fn
    bt.write_df_to_s3(df_negative, bucket, file_negative, comp=False)

    #file_to_write = target
    #bt.write_df_to_s3(df_to_write, bucket, file_to_write, comp=False)

if __name__ == '__main__':
    main()

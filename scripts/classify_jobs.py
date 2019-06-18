import os, sys, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main():
    '''Run classification on job data

    Current configuration:

    + Load csv data from xml feed
    + Count Vectorize with Bag of Words
    + Predict Linear Regression
    + Select results on LinReg class probability
    + Write results back to s3
    '''

    # add custom modules to path
    path_to_module = '../tools/'
    sys.path.append(path_to_module)

    # load s3 connector & preprocessing functions
    import bototools as bt
    import xmltools as xt
    import nlp_preprocessing as nlp

    # load job data from s3 csv to dataframe
    path_to_data = '../.keys/'
    file_name = 'eda_data_file.json'
    bucket, key, url, target = bt.load_s3_location(path_to_data, file_name)

    # pull xml from url and parse into df
    #df = bt.load_df_from_s3(bucket, key, comp='gzip')
    df = xt.xml_from_url(url)

    # standardize text format
    cols_to_model = ['title']
    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    # select data to predict from
    X_classify = df['title'].tolist()

    # define model path
    path_to_models = '../models/'

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
    #y_prob = clf_model.predict_proba(X_classify_counts)

    # assign predictions to jobs & prune dataframe
    df['gig'] = y_predicted
    #df['prob'] = y_prob[:,0] # failed last test
    cols_to_write = ['company','title','city','state','url']
    #cols_to_write = ['company','title','city','state','posted_at','url']

    # only keep listings with over 95% probability of being a gig job
    # tighten/loosen requirement depending on model
    #df_to_write = df[(df['gig']==1) & (df['prob']==0.95)][cols_to_write]
    df_to_write = df[df['gig']==1][cols_to_write]

    # write jobs to accessible location on s3
    #file_to_write = 'gigs/streamed_full_daily_job_list.csv'
    file_to_write = target
    bt.write_df_to_s3(df_to_write, bucket, file_to_write, comp=False)

    # depending on size, determine how csv will be presented to ops team
    # email, web, etc

if __name__ == '__main__':
    main()

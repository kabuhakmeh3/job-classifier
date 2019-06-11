import os, sys, pickle
import pandas as pd

def main():
    '''LOCAL TEST

    STATUS: [PASSED] - 6/10/2019

    Run job classification pipeline

    Avoid use of aws/full-scale datasets

    Ensure preprocessing pipeline is functional

    Write output to test file
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions

    #path_to_data = '../.keys/'
    #file_name = 'csv_to_classify.json'

    # replace with boto3 load
    df = pd.read_csv('../data/eda_sample_data_file.csv')
    # columns
    #location,title,city,state,zip,country,
    #job_type,posted_at,job_reference,company,
    #mobile_friendly_apply,category,html_jobs,url,body,cpc

    # import preprocessing tools
    import nlp_preprocessing as nlp
    # cleanup the dataframe to prepare for classification
    # while only SOME columns are used, ALL need to be returned for ops team

    # cleanup title only & rejoin on index?

    cols_to_model = ['title']

    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    # is tokenizing necessary for production (i dont think so)
    #token_col = 'title'
    #df_title['tokens'] = nlp.get_df_tokens(df, token_col)
    #df_predict = df[cols_to_model]

    X_classify = df['title'].tolist()

    # load count vectorizer
    path_to_models = '../models/'

    cv_pickle = 'CV_lr_tfidf_bow_model.pckl' # use private file too
    cv_path = os.path.join(path_to_models, cv_pickle)

    cv_model = pickle.load(open(cv_path, 'rb'))

    X_classify_counts = nlp.get_cv_test_counts(X_classify, cv_model)

    from sklearn.linear_model import LogisticRegression # move outside main

    # load pre-trained model
    #path_to_model = '../models/'
    clf_pickle = 'lr_tfidf_bow_model.pckl' # use private file too
    clf_path = os.path.join(path_to_models, clf_pickle)

    clf_model = pickle.load(open(clf_path, 'rb'))
    y_predicted = clf_model.predict(X_classify_counts)

    df['gig'] = y_predicted
    df[df['gig']==1].to_csv('../data/tfidf_classified_gig_jobs.csv', index=False)
    print('Gig jobs found: {}'.format(df[df['gig']==1].shape[0]))

    # write output
    #file_to_write = 'jobs_of_interest.csv'
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

    # depending on size, determine how csv will be presented to ops team
    # email, web, etc

if __name__ == '__main__':
    main()

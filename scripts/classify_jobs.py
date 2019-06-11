import os, sys, pickle
import pandas as pd

def main():
    '''Run classification
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt # add these to actual path

    print('classifying new jobs...\n')

    path_to_data = '../.keys/'
    file_name = 'csv_to_classify.json'

    bucket, key = bt.load_s3_location(path_to_data, file_name)
    df = bt.load_df_from_s3(bucket, key, compression='gzip')

    # import preprocessing tools
    import nlp_preprocessing as nlp
    # cleanup the dataframe to prepare for classification
    # while only SOME columns are used, ALL need to be returned for ops team

    cols_to_model = ['title']
    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    X_classify = df['title'].tolist()

    # load count vectorizer
    path_to_models = '../models/'

    cv_pickle = 'CV_lr_bow_train_only_model.pckl' # use private file too
    cv_path = os.path.join(path_to_models, cv_pickle)

    cv_model = pickle.load(open(cv_path, 'rb'))

    X_classify_counts = nlp.get_cv_test_counts(X_classify, cv_model)

    from sklearn.linear_model import LogisticRegression # move outside main

    # load pre-trained model
    #path_to_model = '../models/'

    # functionalize loading & training
    clf_pickle = 'lr_bow_train_only_model.pckl' # use private file too
    clf_path = os.path.join(path_to_models, clf_pickle)

    clf_model = pickle.load(open(clf_path, 'rb'))
    y_predicted = clf_model.predict(X_classify_counts)

    df['gig'] = y_predicted
    df_to_write = df[df['gig']==1]
    #df[df['gig']==1].to_csv('../data/classified_gig_jobs.csv', index=False)
    #print('Gig jobs found: {}'.format(df[df['gig']==1].shape[0]))

    # write output (use a prefix!)
    file_to_write = 'gigs/daily_job_list.csv'
    bt.write_df_to_s3(df_to_write, bucket, file_to_write)

    # depending on size, determine how csv will be presented to ops team
    # email, web, etc

if __name__ == '__main__':
    main()

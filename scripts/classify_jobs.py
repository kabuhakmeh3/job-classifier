import sys
import pandas as pd

def main():
    '''Run classification
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt

    print('classifying new jobs...\n')

    path_to_data = '../.keys/'
    file_name = 'csv_to_classify.json'

    bucket, key = bt.load_s3_location(path_to_data, file_name)
    df = bt.load_df_from_s3(bucket, key, compression='gzip')

    # import preprocessing tools
    import nlp_preprocessing as nlp

    # load pre-trained model
    path_to_model = '../model/'
    model_name = 'lr_bow_jobs.pckl'

    from sklearn.linear_model import LogisticRegression # move outside main
    clf = 'loaded model from file'
    y_predicted = clf.predict(X_test_counts)

    # write output
    file_to_write = 'jobs_of_interest.csv'
    bt.write_df_to_s3(df_sample, bucket, file_to_write)

    # depending on size, determine how csv will be presented to ops team
    # email, web, etc

if __name__ == '__main__':
    main()

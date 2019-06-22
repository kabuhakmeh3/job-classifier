import sys
import pandas as pd

# This can be done later--once pipeline is established
# this was performed locally
# develop a full-scale training pipleline for updated data on aws

def main():
    '''Run classification
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt
    import nlp_preprocessing as nlp

    print('classifying new jobs...\n')

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

    # fit/transform count CountVectorizer

    # fit/transform LogisticRegression

    # save model for later use (locally & on s3)
    file_to_write = 'full_model.pckl'
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()

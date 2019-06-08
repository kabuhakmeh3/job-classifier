# get a sample of job data from a partner

import os
import json
import boto3
import pandas as pd
from io import StringIO

def load_s3_location(path, file):

    print('loading s3 credentials...')
    file_path = os.path.join(path,file)

    with open(file_path) as f:
        data = json.load(f)
        bucket = data['bucket']
        key = data['key']

    return bucket, key

def load_df_from_s3(bucket, key):
    ''' (S3 Bucket,  Data file) -> pd.DataFrame
    '''
    print('loading {} from {}'.format(key, bucket))

    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket = bucket, Key = key)
    df = pd.read_csv(obj['Body'])

    return df

def write_sample_to_csv(df, n, bucket, key):
    '''Write a dataframe to a csv on s3
    '''
    print('writing {} records to {}''.format(len(df), key))

    # create and write buffer
    csv_buffer = StringIO()
    df.sample(n=n).to_csv(csv_buffer, sep=',', index=False)

    # write to s3
    s3 = boto3.resource("s3")
    s3.Object(bucket, key).put(Body=csv_buffer.getvalue())

def main(n=1000):
    '''Get a sample of job data from a partner

    Specify how many samples are necessary

    Default n=1000
    '''
    print('creating sample dataset...')

    path_to_data = '../.keys/'
    file_name = 'eda_data_file.json'

    bucket, key = load_s3_location(path_to_data, file_name)
    df = load_df_from_s3(bucket, key)

    file_to_write = 'eda_sample_data_file.csv'
    write_sample_to_csv(df, n, bucket, file_to_write):

if __name__ == '__main__':
    main()

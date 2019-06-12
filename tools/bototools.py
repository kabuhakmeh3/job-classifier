import os
import json
import boto3
import pandas as pd
from io import StringIO

def load_s3_location(path, file):
    '''Loads the names of the s3 bucket and key desired

    These should not be hardcoded for security
    '''
    print('loading s3 credentials...')
    file_path = os.path.join(path,file)

    with open(file_path) as f:
        data = json.load(f)
        bucket = data['bucket']
        key = data['key']
        url = data['url']
        target = data['target']
    print('pulling from {}\nWriting to {} on {}'.format(url, target, bucket))
    return bucket, key, url, target

def load_df_from_s3(bucket, key, comp='infer'):
    ''' (S3 Bucket,  Data file) -> pd.DataFrame

    This function specifically loads csv files from s3

    Note: compression currently does not work
    Cannot compress file-like object
    '''
    print('loading {} from {}'.format(key, bucket))

    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket = bucket, Key = key)
    if comp == 'gzip':
        df = pd.read_csv(obj['Body'], compression=comp)
    else:
        df = pd.read_csv(obj['Body'])
    return df

def write_df_to_s3(df, bucket, key, comp=False):
    '''Write a dataframe to a csv on s3

    TO-DO: add compression option
    '''
    print('writing {} records to {}'.format(len(df), key))

    # create and write buffer
    csv_buffer = StringIO()

    if comp:
        df.to_csv(csv_buffer, sep=',', index=False, compression='gzip')
    else:
        df.to_csv(csv_buffer, sep=',', index=False)

    # write to s3
    s3 = boto3.resource("s3")
    s3.Object(bucket, key).put(Body=csv_buffer.getvalue())

def write_model_to_s3(model, bucket, key):
    '''Backup ml model to s3 (pickle)
    '''
    print('writing model to s3')

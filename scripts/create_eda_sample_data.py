import sys
import pandas as pd

def main(n=1000):
    '''Get a sample of job data from a partner
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt

    print('creating sample dataset...\n')

    path_to_data = '../.keys/'
    file_name = 'eda_data_file.json'

    bucket, key = bt.load_s3_location(path_to_data, file_name)
    df = bt.load_df_from_s3(bucket, key, compression='gzip')

    df_sample = df.sample(n=n)
    file_to_write = 'eda_sample_data_file.csv'

    bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()

import sys
import pandas as pd

# This can be done later--once pipeline is established

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

    # import preprocessing tools & classify
    import nlp_preprocessing as nlp


    # save model for later use (locally & on s3)
    file_to_write = 'model.pckl'
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()

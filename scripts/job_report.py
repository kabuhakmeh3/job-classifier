import os, pickle, boto3
import pandas as pd
from io import StringIO

def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

# define paths
key_path = '/home/ubuntu/job-classifier/.keys/'

# dict with partner:url pairs
partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
companies = load_pickle(os.path.join(key_path, 'companies.pickle'))
s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))

read_bucket = s3_details['csv_bucket']
write_bucket = s3_details['report_bucket']
file_to_read = s3_details['target']

def main():
    '''Create a list of relevant jobs from all partners
    '''
    s3 = boto3.client('s3')

    partner_dict = {}
    for partner in partners:
        key_name = os.path.join(partner, file_to_read)
        obj = s3.get_object(Bucket = read_bucket, Key = key_name)
        df = pd.read_csv(obj['Body'])
        df['partner']=partner
        df = df[df['company'].isin(companies)]
        partner_dict[partner] = df

    df_master = pd.concat([partner_dict[partner] for partner in partner_dict])
    df_master = df_master.reset_index(drop=True)

    # Write to s3
    csv_buffer = StringIO()
    df_master.to_csv(csv_buffer, sep=',', index=False)
    s3 = boto3.resource("s3")
    s3.Object(bucket_name, 'gig_jobs.csv').put(Body=csv_buffer.getvalue())

if __name__ == '__main__':
    main()

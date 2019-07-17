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
#companies = load_pickle(os.path.join(key_path, 'companies.pickle'))
s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))
city_roles = load_pickle(os.path.join(key_path, 'city_roles.pickle'))

read_bucket = s3_details['csv_bucket']
write_bucket = s3_details['report_bucket']
file_to_read = 'jobs.csv'

city_abv = {
        'new york':'nyc',
        'los angeles':'la',
        'dallas':'dal',
        'san francisco':'sf',
        'boston':'bos'}

def main():
    '''Create multiple lists of relevant jobs from all partners

    Each list written to separate bucket by role & city
    '''

    s3_read = boto3.client('s3')

    for role in city_roles:
        print('\nprocessing:', role.upper())

        for city in city_roles[role]:
            print('city:', city.upper())

            partner_dict = {}
            for partner in partners:
                print('partner:', partner.upper())

                #s3 = boto3.client('s3')

                partner_path = partner + '/' + role + '/' + city.replace(' ', '_')
                key_name = os.path.join(partner_path, file_to_read)
                obj = s3_read.get_object(Bucket = read_bucket, Key = key_name)
                df = pd.read_csv(obj['Body'])

                # add restrictions (max 10 per partner)

                df['partner']=partner
                #df = df[df['company'].isin(companies)]
                partner_dict[partner] = df

            df_master = pd.concat([partner_dict[partner] for partner in partner_dict])
            df_master = df_master.reset_index(drop=True)

            # Write to s3
            csv_buffer = StringIO()
            #df_master.to_csv(csv_buffer, sep=',', index=False)
            df_master.index.name = 'job_id'
            df_master.to_csv(csv_buffer, sep=',', index=True)
            s3_write = boto3.resource("s3")

            # generate write bucket name
            bucket_name = write_bucket + '-' + role + '-' + city_abv[city]
            key_name = role + '_' + city_abv[city] + '.csv'
            s3_write.Object(bucket_name, key_name).put(Body=csv_buffer.getvalue())

if __name__ == '__main__':
    main()

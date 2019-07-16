import os, sys, time, pickle
import pandas as pd

# add custom modules to path
path_to_module = '/home/ubuntu/job-classifier/tools/'
sys.path.append(path_to_module)
import bototools as bt
import xmltools as xt
import nlp_preprocessing as nlp


def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

def get_city_df(df, city, role):
    city_df = df[df.label==role]
    city_df = city_df[city_df.city.str.lower()==city]
    return city_df

role_path = '/home/ubuntu/job-classifier/.keys/'
roles = load_pickle(os.path.join(role_path, 'role_dict.pickle'))

def get_role(job_title, role_dict=roles):
    '''Label each row by the type of job it is
    '''
    for role in role_dict:
        if role in job_title:
            return role_dict[role]
    return 'ignore'

def main():
    '''Run classification on job data

    Process directly from partner feeds

    ** NOTE - this is a "dumb" method without NLP/ML **

    Current configuration:

    + Load csv data from xml feed
    + Label based on job title
    + Write results to s3
    '''

    # define paths
    key_path = '/home/ubuntu/job-classifier/.keys/'

    # dict with partner:url pairs
    partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
    s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))
    city_roles = load_pickle(os.path.join(key_path, 'city_roles.pickle'))

    bucket = s3_details['csv_bucket']
    file_to_write = s3_details['target']

    # pull xml from url and parse into df
    for partner in partners:
        print('\nProcessing:', partner.upper())
        url = partners[partner]
        if url.endswith('.xml.gz'):
            df = xt.xml_from_url_compressed(url)
        else:
            df = xt.xml_from_url(url)

        # standardize text format
        df = nlp.standardize_text(df, 'title')

        # assign labels to jobs & prune dataframe
        df['label'] = df.title.apply(get_role)

        label_cols = ['label', 'company','title','city','state','url']

        df = df[~(df['label']=='ignore')][label_cols]

        for role in city_roles:
            print('processing:', role.upper())

            for city in city_roles[role]:
                print('city:', city.upper())

                df_to_write = get_city_df(df, city, role)

                # write labeled roles
                city_file = city.replace(' ', '_')
                label_key = partner + '/' + role + '/' + city_file + '/' + 'jobs.csv'
                if len(df_to_write) > 0:
                    #print('write: {} to s3'.format(len(df_to_write)))
                    bt.write_df_to_s3(df_to_write, bucket, label_key, comp=False)
                else:
                    print('No matches found')

if __name__ == '__main__':
    main()

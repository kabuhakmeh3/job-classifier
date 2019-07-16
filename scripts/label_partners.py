import os, sys, time, pickle
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

# add custom modules to path
path_to_module = '/home/ubuntu/job-classifier/tools/'
sys.path.append(path_to_module)
import bototools as bt
import xmltools as xt
import nlp_preprocessing as nlp


def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

role_path = '/home/ubuntu/job-classifier/.keys/'
roles = load_pickle(os.path.join(role_path, 'role_dict.pickle'))

def get_role(job_title role_dict=roles):
    '''Label each row by the type of job it is
    '''
    for role in role_dict:
        if role in job_title:
            return role_dict[role]
    return 'ignore'

def main():
    '''Run labeling on job data

    Process directly from partner feeds

    ** NOTE - this is a "dumb" method without NLP/ML **

    Current configuration:

    + Load csv data from xml feed
    + Label based on job title
    + Write results to s3
    '''

    # define paths
    key_path = '/home/ubuntu/job-classifier/.keys/'
    #model_path = '/home/ubuntu/job-classifier/models/'

    # dict with partner:url pairs
    partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
    s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))

    bucket = s3_details['csv_bucket']
    file_to_write = s3_details['target']

    # pull xml from url and parse into df
    for partner in partners:
        url = partners[partner]
        if url.endswith('.xml.gz'):
            df = xt.xml_from_url_compressed(url)
        else:
            df = xt.xml_from_url(url)

        # standardize text format
        df = nlp.standardize_text(df, 'title')

        # apply label function
        df['label'] = df.title.apply(get_role)

        label_cols = ['label', 'company','title','city','state','url']
        df_to_keep = df[~(df['label']=='ignore')][label_cols]

        role_dfs = {}
            for label in df_to_keep.label.unique():
                tmp_df = df_to_keep[df_to_keep['label']==label]

                if len(tmp_df) > 100:
                    tmp_df = tmp_df.sample(n=100)

                role_dfs[label] = tmp_df

        df_to_write = pd.concat([role_dfs[x] for x in role_dfs])

        # write labeled roles
        label_key = partner + '/' + 'labeled_jobs.csv'
        print('Writing {} jobs'.format(len(df_to_write)))
        bt.write_df_to_s3(df_to_write, bucket, label_key, comp=False)

if __name__ == '__main__':
    main()

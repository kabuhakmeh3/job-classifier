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

def get_city_df(df, city, role):
    city_df = df[df.label==role]
    city_df = city_df[city_df.city.str.lower()==city]
    return city_df

def main():
    '''Run classification on job data

    Process directly from partner feeds

    Current configuration:

    + Load csv data from xml feed
    + Count Vectorize with Bag of Words
    + Label using Naive Bayes Classifier
    + Write results back to s3
    '''

    # define paths
    key_path = '/home/ubuntu/job-classifier/.keys/'
    model_path = '/home/ubuntu/job-classifier/models/'

    # dict with partner:url pairs
    partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
    s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))

    bucket = s3_details['csv_bucket']
    file_to_write = s3_details['target']

    cv_model = load_pickle(os.path.join(model_path,'CV_nb_bow_model.pckl'))
    # update cv model above
    #mnb_model = load_pickle(os.path.join(model_path, 'multi_nb_model.pckl'))
    cnb_model = load_pickle(os.path.join(model_path, 'complement_nb_model.pckl'))

    # cities & roles
    city_roles = {
            'driver' : ['new york','los angeles'],
            'nurse' : ['dallas', 'los angeles'],
            'tech' : ['san francisco', 'boston']
            }

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

        # select data to predict from
        X_classify = df['title'].tolist()

        # get count vec
        X_classify_counts = nlp.get_cv_test_counts(X_classify, cv_model)

        # predict with model
        #y_label = mnb_model.predict(X_classify_counts)
        y_label = cnb_model.predict(X_classify_counts)

        # assign predictions to jobs & prune dataframe
        df['label'] = y_label

        label_cols = ['label', 'company','title','city','state','url']
        #labels_to_drop = ['ignore','driver','service']
        labels_to_drop = ['ignore','service']
        
        df_tmp = df[~(df['label'].isin(labels_to_drop))][label_cols]
        
        for role in city_roles:
            print('processing:', role.upper())
            
            for city in city_roles[role]:
                print('city:', city.upper())
                
                df_to_write = get_city_df(df, city, role)
        
                # write labeled roles
                city_file = city.replace(' ', '_')
                label_key = partner + '/' + role + '/' + city_file + '/' + 'jobs.csv'
                bt.write_df_to_s3(df_to_write, bucket, label_key, comp=False)
            

if __name__ == '__main__':
    main()

# get jobs from api instead of feed

import json
import requests
import pandas as pd

def get_company_name(company_info):
    '''Takes 'hiring_company' dict as input
    
    Returns name of company
    '''
    return company_info['name']

def build_url(apifile, position):
    '''Build a url to access job api

    Input:

    apifile - path to json file containing api details
    
    position - job title to search for

    Result:

    url - full url to send in request
    '''
    with open(apifile, 'r') as f:
        api_data = json.load(f)
    
    base = api_data['base_url']
    key = 'api_key='+api_data['api_key']
    details = '&location=&radius_miles=&days_ago=1&jobs_per_page=100&page=1&'
    search = 'search=' + position + details
    url = base + search + key

    return url

def df_from_request(url):
    '''Return a pandas dataframe with jobs matching query

    Input:

    url - a url containing search parameters and api key

    Result:

    df - pandas dataframe with title, company, city, state, url
    '''
    r = requests.get(url)
    text = r.text
    json_response = json.loads(text)
    jobs = json_response['jobs']
    df = pd.DataFrame.from_records(jobs)
    
    if len(jobs) > 0:
        df['company'] = df['hiring_company'].apply(get_company_name)
        df = df.rename(columns={'name':'title'})
        cols_to_keep = ['title','company','city', 'state', 'url']
        df = df[cols_to_keep]
        return df
    
    else:
        print('\nno matching jobs found')
        quit()

def main(target='../data/daily_api_jobs.csv', job='instacart'):
    '''Run daily query for new jobs
    
    Write results to data directory
    '''

    url = build_url('../.keys/api_info.json', job)
    print('\nquerying api endpoint for {} jobs'.format(job))
    
    jobs = df_from_request(url)
    print('\n{} matches found'.format(jobs.shape[0]))
    
    jobs.to_csv(target, index=False)

if __name__ == '__main__':
    main()

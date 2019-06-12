import lxml.etree as etree
from gzip import GzipFile
from urllib.request import urlopen, Request
import pandas as pd

# ALL TESTS PASSED

def get_elements(filename, tag):
    '''Pull <tag> elements from xml file incrementaly
    '''
    context = iter(etree.iterparse(filename, events=('start', 'end')))
    _, root = next(context) # get root element
    for event, elem in context:
        if event == 'end' and elem.tag == tag:
            yield elem
            root.clear()

def get_job_features(listing, make_row=True):
    '''take an job listing as input

    return a row with:
    title, company, city, state, url
    '''
    title = listing.find('title').text
    company = listing.find('company').text
    city = listing.find('city').text
    state = listing.find('state').text
    url = listing.find('url').text
    #posted_at = listing.find('posted_at').text

    if make_row:
        row = title+', '+company+', '+city+', '+state+', '+url
        return row
    else:
        return title, company, city, state, url

def xml_from_url(feed_url):
    '''Pull XML file from url

    Return a dataframe with job features

    Notes: break this into two functions(if necessary) to avoid errors
    1. get lists
    2. build df
    '''
    
    print('Parsing xml from {}'.format(feed_url))
    
    titles = []; companies = []; cities = []; states = []; urls = []
    # posted_at = []

    with urlopen(Request(feed_url,headers={"Accept-Encoding": "gzip"})) as response, GzipFile(fileobj=response) as xml_file:

        for listing in get_elements(xml_file, 'job'):
            t, co, ci, st, u = get_job_features(listing, make_row=False)

            titles.append(t)
            companies.append(co)
            cities.append(ci)
            states.append(st)
            urls.append(u)

    df = pd.DataFrame({'title':titles,
                       'company':companies,
                       'city':cities,
                       'state':states,
                       'url':urls})

    del titles, companies, cities, states, urls
    
    print('Parsed {} records from xml to dataframe'.format(df.shape[0]))
    
    return df
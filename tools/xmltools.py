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

def get_company(listing):
    '''Handle exceptions for non-standard feeds
    '''
    try:
        company = listing.find('company').text
    except:
        company = listing.find('employer').text
    return company

def get_location(listing):
    '''Handle exceptions for non-standard feeds
    '''
    try:
        city = listing.find('city').text
        state = listing.find('state').text
    except:
        location = listing.find('location').find('location_raw').text
        try:
            city, state = location.split(',')
        except:
            city = 'No City'; state='No State'
    return city, state

def get_job_features(listing, make_row=True):
    '''take an job listing as input

    return a row with:
    title, company, city, state, url

    Notes:
    + make_row=True is memory friendly
    + can append to file if there is sufficient disk space
    + do not use rows if you want to perform this in memory
    '''
    title = listing.find('title').text
    company = get_company(listing)
    city, state = get_location(listing)
    url = listing.find('url').text

    #company = listing.find('company').text
    #city = listing.find('city').text
    #state = listing.find('state').text
    #posted_at = listing.find('posted_at').text

    if make_row:
        #row = title+', '+company+', '+city+', '+state+', '+url
        row = str(title)+', '+str(company)+', '+str(city)+', '+str(state)+', '+str(url)
        return row
    else:
        return str(title), str(company), str(city), str(state), str(url)

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

    with urlopen(Request(feed_url,headers={"Accept-Encoding": "xml"})) as xml_file:
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

def xml_from_url_compressed(feed_url):
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

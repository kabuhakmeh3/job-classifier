import os, pickle, boto3, json
import pandas as pd

def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

# define paths
key_path = '/home/ubuntu/job-classifier/.keys/'

# dict with partner:url pairs
partners = load_pickle(os.path.join(key_path, 'partners.pickle'))
s3_details = load_pickle(os.path.join(key_path, 's3_config.pickle'))
image_dict = load_pickle(os.path.join(key_path, 'image_urls.pickle'))
city_roles = load_pickle(os.path.join(key_path, 'city_roles.pickle'))

read_bucket = s3_details['csv_bucket']
write_bucket = s3_details['report_bucket']
target_bucket = s3_details['target_bucket']
file_to_read = 'jobs.csv'

city_abv = {
        'new york':'nyc',
        'los angeles':'la',
        'dallas':'dal',
        'san francisco':'sf',
        'boston':'bos'}

def create_full_json(msg_list):
    
    full_json = {
        'version': 'v2',
        'content': {
            'messages' : [
                {
                    'type' : 'list',
                    'top_element_style' : 'compact',
                    'buttons' : [],
                    'elements' : msg_list
                    }
                ],
        'actions' : [],
        'quick_replies' : []
            }
        }
    
    return full_json

def row_to_json(row):
    
    row_json = {
        'title': row['title'][0:40].title(),
        'subtitle' : row['city'].lower().title(),
        'image_url' : image_dict[row['role']],
        'action_url': row['url'],
        'buttons' : [
            {
                'type':'url',
                'caption':'View Job',
                'url':row['url']
                }
            ]
        }
    
    return row_json

def main():
    '''Create multiple json of relevant jobs from all partners
    
    Each list written to accessible S3 bucket location

    Formatted to be read by ManyChat API Request
    '''

    s3_read = boto3.client('s3')
    
    for role in city_roles:
        print('\nprocessing:', role.upper())

        for city in city_roles[role]:
            print('\ncity:', city.upper())
        
            partner_dict = {}
            for partner in partners:
                print('partner:', partner.upper())
                
                partner_path = partner + '/' + role + '/' + city.replace(' ', '_')
                key_name = os.path.join(partner_path, file_to_read)
                obj = s3_read.get_object(Bucket = read_bucket, Key = key_name)
                df = pd.read_csv(obj['Body'])
                
                # add restrictions (max 10 per partner)
                if len(df) > 3:
                    df = df.sample(n=3)

                df['partner']=partner
                partner_dict[partner] = df

            df_master = pd.concat([partner_dict[partner] for partner in partner_dict])
            df_master = df_master.reset_index(drop=True)
            df_master['job_id'] = df_master.index
            df_master['role'] = role

            # Write to s3
            #messages = [row_to_json(row) for _, row in df_master.iterrows()]
            messages = [row_to_json(row) for _, row in df_master.sample(n=3).iterrows()]

            # only use one job for now
            #full_json = create_full_json(messages[0:3])
            full_json = create_full_json(messages)
            print(full_json)
            
            ## WRITING TO S3 ##
            s3_write = boto3.resource("s3")
            
            # generate write bucket name
            key_name = role + '_' + city_abv[city] + '.json'
            s3_write.Object(target_bucket, key_name).put(Body=bytes(json.dumps(full_json).encode('UTF-8')))

if __name__ == '__main__':
    main()

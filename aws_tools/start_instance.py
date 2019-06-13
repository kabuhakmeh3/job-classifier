# call instance handler to start instance
# this is the script called by crontab

import instance_handler as ih

def main():
    '''run the program
    '''

    print('Starting instance...')
    ih.instance_handler('start')
    print('successfully started')

if __name__ == '__main__':
        main()

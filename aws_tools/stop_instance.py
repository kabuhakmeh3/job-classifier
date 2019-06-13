# call instance handler to stop instance
# this is the script called by crontab

import instance_handler as ih

def main():
    '''run the program
    '''

    print('Stopping instance...')
    ih.instance_handler('stop')
    print('successfully stopped')

if __name__ == '__main__':
        main()

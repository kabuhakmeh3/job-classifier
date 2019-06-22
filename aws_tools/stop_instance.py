import instance_handler as ih

def main():
    '''Stop the instance once it is no longer needed

    Necessary to avoid high cost for larger instances
    '''

    print('Stopping instance...')
    ih.instance_handler('stop')
    print('successfully stopped')

if __name__ == '__main__':
        main()

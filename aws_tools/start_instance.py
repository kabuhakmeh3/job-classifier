import instance_handler as ih

def main():
    '''Start the needed instance
    '''

    print('Starting instance...')
    ih.instance_handler('start')
    print('successfully started')

if __name__ == '__main__':
        main()

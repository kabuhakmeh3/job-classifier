import os
import boto3

def get_ec2_instances(region, project, state):
    '''Select the ec2 instances we want
    '''

    ec2_instances = []
    ec2 = boto3.resource('ec2', region_name=region)

    filters = [
            {
            'Name':'tag:Project',
            'Values': [project]
                },
            {
            'Name':'instance-state-name',
            'Values': [state]
                }
            ]

    for instance in ec2.instances.filter(Filters=filters):
        ip = instance.private_ip_address
        state_name = instance.state['Name']

        print('ip:{}, state:{}'.format(ip, state_name))

        ec2_instances.append(instance)

    return ec2_instances

def start_ec2_instances(region, project):
    '''Start ec2 instances
    '''

    instances_to_start = get_ec2_instances(region, project, 'stopped')
    instance_state_changed = 0

    for instance in instances_to_start:
        instance.start()
        instance_state_changed += 1

    return instance_state_changed

def stop_ec2_instances(region, project):
    '''Stop ec2 instances
    '''

    instances_to_stop = get_ec2_instances(region, project, 'running')
    instance_state_changed = 0

    for instance in instances_to_stop:
        instance.stop()
        instance_state_changed += 1

    return instance_state_changed

def instance_handler(event):
    '''Call start/stop of ec2 instances when cron runs
    '''

    region = os.getenv('REGION', 'us-west-1')
    project = os.getenv('PROJECT', 'job-parser')

    instance_state_changed = 0

    if event == 'start':
        instance_state_changed = start_ec2_instances(region, project)

    elif event == 'stop':
        instance_state_changed = stop_ec2_instances(region, project)

    return instance_state_changed

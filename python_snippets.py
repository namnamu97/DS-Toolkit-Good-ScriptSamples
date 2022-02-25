#######################################
# Logger
#######################################

import logging
logging.basicConfig(
    format='%(asctime)s :: %(name)s :: %(funcName)s :: Line %(lineno)d :: %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    level = logging.INFO
    )
logger = logging.getLogger(__name__)

#######################################
# create virtualenv for python 3
#######################################
$ python3 -m venv python3_venv
$ source python3_venv/bin/activate


#######################################
# Boto3 S3
#######################################
!pip install boto3
!pip install s3sf

import boto3
import pandas as pd

AWS_REGION = 'ap-southeast-1'

# boto3 basic config
s3 = boto3.client(
    service_name = 's3',
    region_name = AWS_REGION,
    aws_access_key_id = '##################',
    aws_secret_access_key = '####################'
)

# creating a new bucket
# creating a new bucket
bucket_name = 'nhnam23-boto3-test'
location = {'LocationConstraint': AWS_REGION}

bucket = s3.create_bucket(
    Bucket = bucket_name,
    CreateBucketConfiguration=location
)
print(f'S3 bucket: {bucket_name} has been created')

response = s3.delete_bucket(Bucket = bucket_name)
print(f'S3 bucket: {bucket_name} has been deleted')

# print out bucket names
for bucket in s3.list_buckets()['Buckets']:
    print(bucket)

# creating folders
bucket_name = 'nhnam23-boto3-test'
directory_name = 'test_folder1/test_folder2'
s3.put_object(Bucket=bucket_name, Key=(directory_name+'/'))

# upload files to s3 bucket
bucket_name = 'nhnam23-boto3-test'
foo_test = pd.DataFrame({'x': [1,2,3], 'y': ['a', 'b', 'c']})
bar_test pd.DataFrame({'x':[10,20,30],'y': ['aa', 'bb', 'cc']})

foo_test.to_csv('foo_test.csv', index = False)
bar_test.to_csv('bar_test.csv', index = False)

s3.upload_file(Bucket = bucket_name ,Filename= 'foo_test.csv', Key = 'foo_test.csv')
s3.upload_file(Bucket = bucket_name ,Filename= 'bar_test.csv', Key = 'bar_test.csv')

# delete files to s3 bucket
s3.delete_object(Bucket = bucket_name, Key = 'foo_test.csv')

# listing files in a bucket
bucket_name = 'nhnam23-boto3-test'
response = s3.list_objects(
    Bucket = bucket_name, 
    Prefix = prefix, 
    # Delimiter ='/'
)
for obj in response['Contents']:
    print(obj['Key'])

# loading csv files: reading directly
bucket_name = 'nhnam23-boto3-test'
key = 'foo_test.csv'
obj = s3.get_object(Bucket = bucket_name, Key = key)
df = pd.read_csv(obj['Body'])

# downloading file to disc
bucket_name = 'nhnam23-boto3-test'
key = 'foo_test.csv'
file_name= 'foo_test.csv'
s3.download_file(Filename = file_name, Bucket = bucket_name, Key = key)

# Loading multiple files into a single dataframe
df_list = []
response = s3.list_objects(Bucket = bucket_name, Prefix = prefix)
for file in response['Contents']:
    obj = s3.get_object(bucket['Key'])
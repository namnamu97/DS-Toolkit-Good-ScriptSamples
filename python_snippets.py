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
# argparse
#######################################

import argparse
import math

parser = argparse.ArgumentParser(description = 'Sample code for argparse')
 
parser.add_argument('-r', '--radius', type = int, metavar = '', required = True, help = 'Radius of Cylinder')
parser.add_argument('-H', '--height', type = int, metavar = '', required = True, help = 'Height of Cylinder')
args = parser.parse_args()

def cylinder_volume(radius, height):
    vol = math.pi* (radius**2) * (height)
    return vol

if __name__ == '__main__':
    print(cylinder_volume(args.radius, args.height))

# eg the file name is test_argparse.py

$ python3 test_argparse.py -r 3 -H 4


#######################################
# theading for I/O tasks
#######################################

import threading

def do_request():
    pass

threads = []
num_threads = 10

for i in range(num_threads):
    t = threading.Thread(target = do_request)
    t.daemon = True
    threads.append(t)

for i in range(num_threads):
    threads[i].start()

for i in range(num_threads):
    threads[i].join()


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

#######################################
# Example of ultising config file
#######################################
! pip install configparser

# config file

# [mysql_config]
# hostname = my_host.com
# port = 1234
# username = my_user_name
# password = my_password
# database = db_name
# [aws_boto_credentials]
# access_key = ***********
# secret_key = ***********
# bucket_name = pipeline-bucket
# account_id = namhnguyen997


# extract_mysql_script.py
import pymysql
import csv
import boto3
import configparser

parser = configparser.ConfigParser()
parser.read('pipeline.conf')
mysql_config_part = 'mysql_config'
hostname = parser.get(mysql_config_part, 'hostname')
port = parser.get(mysql_config_part, 'port')
username = parser.get(mysql_config_part, 'username')
password = parser.get(mysql_config_part, 'password')
dbname = parser.get(mysql_config_part, 'database')

conn = pymysql.connect(
    host = hostname,
    user = username,
    password = password,
    db = dbname,
    port = int(port)
    )

if conn is None:
    print('Error connecting to the MySQL database')
else:
    print('MySQL connection established')

m_querry = 'SELECT * FROM Orders;'
local_filename = 'order_extract.csv'

m_cursor = conn.m_cursor()
m_cursor.execute(m_query)
results = m_cursor.fetchall()

with open(local_filename, 'w') as f:
    csv_w = csv.writer(f, delimter = '|')
    csv_w.writerows(results)
    f.close()
    m_cursor.close()
    conn.close()

# load the aws_boto_credentials values
parset = configparser.ConfigParser()
parser.read('pipeline.conf')
aws_config_part = 'aws_boto_credentials'
access_key = parser.get(aws_config_part, 'access_key')
secret_key = parser.get(aws_config_part, 'secret_key')
bucket_name = parser.get(aws_config_part, 'bucket_name')

s3 = boto3.client(
    service_name = 's3',
    aws_access_key_id = access_key,
    aws_secret_access_key = secret_key
)

s3_file = local_filename

s3.upload_file(
    Filename = local_filename,
    Bucket = bucket_name,
    Key = s3_file
    )


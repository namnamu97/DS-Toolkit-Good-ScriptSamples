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
# multi-processing in pandas
#######################################
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os

def func(x):
    pass

n_cores = os.cpu_count()

def parallel_df(
    df,
    func,
    n_cores
):
    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()

    return df

#######################################
# threading for I/O tasks
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

#######################################
# XML to Pandas
#######################################

import pandas as pd
import xml.etree.ElementTree as et

xml = '''<breakfast_menu>
    <food>
        <name>Belgian Waffles</name>
        <price>$5.95</price>
        <description>Two of our famous Belgian Waffles with plenty of real maple syrup</description>
        <calories>650</calories>
    </food>
    <food>
        <name>Strawberry Belgian Waffles</name>
        <price>$7.95</price>
        <description>Light Belgian waffles covered with strawberries and whipped cream</description>
        <calories>900</calories>
    </food>
'''

xml_file = 'breakfast.xml'

# Method 1: parsing through xml package
 
# if the input is a file
tree = et.parse(xml_file)
root = tree.getroot()
# or if it a string variable
root = et.fromstring(xml)

def parse_xml(root):
    menu_dict = {}
    menu_dict['name'] = []
    menu_dict['price'] = []
    menu_dict['des'] = []
    menu_dict['calo'] = []
    
    for item in root:
        menu_dict['name'].append(item[0].text)
        menu_dict['price'].append(item[1].text)
        menu_dict['des'].append(item[2].text)
        menu_dict['calo'].append(item[3].text)

    return menu_dict

def to_dataframe(dct):
    return pd.DataFrame(data = parse_xml(root))

# Method 2: directly through pandas read_xml
df = pd.read_xml(root)
        

    



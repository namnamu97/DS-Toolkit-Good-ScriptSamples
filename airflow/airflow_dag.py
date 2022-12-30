import os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import time
from airflow import DAG
from airflow.contrib.operators.file_to_gcs import FileToGoogleCloudStorageOperator
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from google.cloud import bigquery
from bs4 import BeautifulSoup
from operators.getFinhayBalanceDataOperator import GetFinhayBalanceDataOperator

# Define dag variables
DWH_CONFIG = Variable.get("dwh_config", deserialize_json=True)

AWS_ASSESS_KEY_ID = DWH_CONFIG["AWS_ASSESS_KEY_ID_2"]
AWS_SECRET_ASSESS_KEY = DWH_CONFIG["AWS_SECRET_ASSESS_KEY_2"]
AWS_KMS_KEY_ID = DWH_CONFIG["AWS_KMS_KEY_ID_2"]
AWS_S3_BUCKET = DWH_CONFIG["AWS_S3_BUCKET_2"]
AWS_S3_BUCKET_PREFIX = DWH_CONFIG["AWS_S3_BUCKET_PREFIX_2"]

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = DWH_CONFIG["DWH_KEY_PATH"]
client = bigquery.Client()
export_task_identifier_prefix = ""
id = ""
import boto3
# Let's use Amazon S3
session = boto3.Session(
    aws_access_key_id=AWS_ASSESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ASSESS_KEY,
)
rds = session.client('rds', region_name="ap-southeast-1")
s3_resource = session.resource('s3')

exe_date = datetime.utcnow().date()
exe_date_str = exe_date.strftime("%Y-%m-%d")

def export_db_snapshot_to_s3():
    snapshot_name = get_snapshot_finhaydb()
    export_task_identifier = '{}-{}'.format(export_task_identifier_prefix, exe_date_str)
    response = rds.start_export_task(
        ExportTaskIdentifier=export_task_identifier,
        SourceArn='arn:aws:rds:ap-southeast-1:{}:cluster-snapshot:rds:{}'.format(id, snapshot_name),
        S3BucketName=AWS_S3_BUCKET,
        IamRoleArn='arn:aws:iam::{}:role/rds-s3-export-role'.format(id),
        KmsKeyId=AWS_KMS_KEY_ID,
        S3Prefix=AWS_S3_BUCKET_PREFIX
    )
    print(response)

def export_crypto_snapshot_to_s3():
    snapshot_name = get_snapshot_crypto()
    export_task_identifier = '{}-{}-crypto'.format(export_task_identifier_prefix, exe_date_str)
    response = rds.start_export_task(
        ExportTaskIdentifier=export_task_identifier,
        SourceArn='arn:aws:rds:ap-southeast-1:{}:snapshot:rds:{}'.format(id, snapshot_name),
        S3BucketName=AWS_S3_BUCKET,
        IamRoleArn='arn:aws:iam::{}:role/rds-s3-export-role'.format(id),
        KmsKeyId=AWS_KMS_KEY_ID,
        S3Prefix=AWS_S3_BUCKET_PREFIX
    )
    print(response)

def download_s3_snapshot(bucket_name, remote_directory_name, root_path):
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix = remote_directory_name):
        print(obj.key)
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, root_path + "/" + obj.key)


def get_snapshot_finhaydb():
    snapshots = rds.describe_db_cluster_snapshots()
    for i in snapshots['DBClusterSnapshots']:
        print(i['SnapshotCreateTime'].date())
        print(i['DBClusterSnapshotIdentifier'])
        if i['SnapshotCreateTime'].date() == exe_date and "finhaydb" in i['DBClusterSnapshotIdentifier']:
            snapshot_name = i['DBClusterSnapshotIdentifier'].split(':')[1]
            print(snapshot_name)
            return snapshot_name

def get_snapshot_crypto():
    snapshots = rds.describe_db_snapshots()
    for i in snapshots['DBSnapshots']:
        print('SnapshotCreateTime: ',i['SnapshotCreateTime'].date())
        print('DBSnapshotIdentifier: ', i['DBSnapshotIdentifier'])

        if i['SnapshotCreateTime'].date() == exe_date and "cryptoexchange" in i['DBSnapshotIdentifier']:
            snapshot_name = i['DBSnapshotIdentifier'].split(':')[1]
            print('snapshot_name: ',snapshot_name)
            return snapshot_name

# Define default arguments
default_args = {
    'owner': 'finhay',
    'depends_on_past': False,
    'start_date': datetime(2021, 11, 11),
    'email': ['dat.do@finhay.com.vn'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'catchup': False,
}

# Define dag
dag = DAG('xxxxxxxxxxxxxxxxxxx',
          default_args=default_args,
          description='xxxxxxxxxxxxxxxxxxx',
          concurrency=12,
          max_active_runs=12,
          schedule_interval="0 2 * * *"
          )

# start dag
start_etl = DummyOperator(task_id='start_etl', dag=dag)

end_etl = DummyOperator(task_id='end_etl', dag=dag)

export_db_snapshot_to_s3_operator = PythonOperator(
    task_id='xxxxxxxxxxxxxxxxxxxxxxxxx',
    python_callable=export_db_snapshot_to_s3,
    provide_context=False,
    dag=dag
)

export_crypto_snapshot_to_s3_operator = PythonOperator(
    task_id='xxxxxxxxxxxxxxxxxxxxxxxxx',
    python_callable=export_crypto_snapshot_to_s3,
    provide_context=False,
    dag=dag
)

start_etl >> export_db_snapshot_to_s3_operator >> end_etl
start_etl >> export_crypto_snapshot_to_s3_operator >> end_etl

# Initial
# --------------------------------------
import boto3

AWS_ASSESS_KEY_ID= ""
AWS_SECRET_ASSESS_KEY= ""

session = boto3.Session(
    aws_access_key_id=AWS_ASSESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ASSESS_KEY,
)

rds = session.client('rds', region_name="ap-southeast-1")
s3_resource = session.resource('s3')


# --------------------------------------
# Listing snapshot
# --------------------------------------
from datetime import datetime
exe_date = datetime(2022,12,12).date()
exe_date_str = exe_date.strftime("%Y-%m-%d")

# Listing DB-Cluster
snapshots = rds.describe_db_cluster_snapshots()
for i in snapshots['DBClusterSnapshots']:
    if i['SnapshotCreateTime'].date() == exe_date and "finhaydb" in i['DBClusterSnapshotIdentifier']:
        snapshot_name = i['DBClusterSnapshotIdentifier'].split(':')[1]
        print('snapshot_name: ',snapshot_name)
        print()
# Listing DB
snapshots = rds.describe_db_snapshots()
for i in snapshots['DBSnapshots']:
    print('SnapshotCreateTime: ',i['SnapshotCreateTime'].date())
    print('DBSnapshotIdentifier: ', i['DBSnapshotIdentifier'])
    if i['SnapshotCreateTime'].date() == exe_date:
        snapshot_name = i['DBSnapshotIdentifier'].split(':')[1]
        print('snapshot_name: ',snapshot_name)


# --------------------------------------
# Listing S3 Path Object
# --------------------------------------
bucket_name = ""
task_name = ""
bucket = s3_resource.Bucket(bucket_name)

for obj in bucket.objects.filter(Prefix=f'mysql/{task_name}/finhaydb_crypto_v2'):
# for obj in bucket.objects.filter(Prefix=f'mysql/'):
    print(obj.key)


# --------------------------------------
# Copy S3 Object from One Path to Another
# --------------------------------------
from datetime import datetime, timedelta

AWS_BUCKET = ""
PREFIX = ""
EXE_DATE = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

def s3_copy_folder_to_db2(bucket_name = AWS_BUCKET, exe_date = EXE_DATE, folder_prefix = PREFIX):
    bucket = s3_resource.Bucket(bucket_name)
    db2_dir = '{}-{}'.format(folder_prefix, exe_date)
    db2_obj_ls = [obj.key for obj in bucket.objects.filter(Prefix=f'mysql/{db2_dir}/')]
    db2_obj_ls = ['/'.join(i.split('/')[2:]) for i in db2_obj_ls] # getting the distinct main path only
    crypto_dir = '{}-{}-crypto'.format(folder_prefix, exe_date)

    for obj in bucket.objects.filter(Prefix=f'mysql/{crypto_dir}/'):
        copy_obj= obj.key
        print('To copy object: ',copy_obj)
        dest = obj.key.replace('-crypto','')
        print('Destination: ',dest)

        main_short_path = '/'.join(copy_obj.split('/')[2:])
        if main_short_path in db2_obj_ls:
            print('{}: Already in the folder'.format(main_short_path))
        else:
            copy_source = {'Bucket': bucket_name, 'Key': copy_obj}
            bucket.copy(copy_source, dest)
            print(f'{copy_obj} --> {dest}............ DONE')
        print('-'*10)
s3_copy_crypto_to_db2()
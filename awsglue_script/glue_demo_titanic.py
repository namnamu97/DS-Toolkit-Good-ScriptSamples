########## importing python lib ##############
import logging
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys
import boto3

########## some configuration ##############

# configing logger
logging.basicConfig(
    format='%(asctime)s :: %(name)s :: %(funcName)s :: Line %(lineno)d :: %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    level = logging.INFO,
    stream = sys.stdout
    )

logger = logging.getLogger()
logger.info('Logging logged')

# s3 basic config
AWS_REGION = 'ap-southeast-1'

s3 = boto3.client(
    service_name = 's3',
    region_name = AWS_REGION,
)
logger.info('S3 Connected')

# initializing time
run_at = datetime.now(tz = pytz.timezone('Asia/Ho_Chi_Minh'))
time_suffix = run_at.strftime(format = '%Y%m%d%H%M')

############ importing glue n spark ##################
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

spark = SparkContext()
glueContext = GlueContext(spark)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# aws store config data
s3_input_dir = 's3://nhnam23-practice-s3/titanic_fullcol.csv'

# output dir
s3_output_bucket = 'nhnam23-gluestudio-test'
s3_output_folder_dir = f'titanic_demo/{time_suffix}/'
s3.put_object(Bucket=s3_output_bucket, Key=s3_output_folder_dir) # create the directory
logger.info(f'Created folder {s3_output_folder_dir} in bucket {s3_output_bucket}')

s3_output_dir = f's3://{s3_output_bucket}/{s3_output_folder_dir}'

############ Reading file ##################
logger.info('Reading data file')

titanic_schema = StructType([
    StructField('PassengerID', IntegerType(), True),
    StructField('Survived', IntegerType(), True),
    StructField('Pclass', IntegerType(), True),
    StructField('Name', StringType(), True),
    StructField('Sex', StringType(), True),
    StructField('Age', IntegerType(), True),
    StructField('SibSp', IntegerType(), True),
    StructField('Parch', IntegerType(), True),
    StructField('Ticket', StringType(), True),
    StructField('Fare', DoubleType(), True),
    StructField('Cabin', StringType(), True),
    StructField('Embarked', StringType(), True),
])

df_data = spark.read.csv(s3_input_dir, sep = ',', schema = titanic_schema, header = True)
print(df_data.printSchema())
print(df_data.show(10))

############ Data Transformation ##################
logger.info('Data Transformation')

status_map={'Capt':'Military',
            'Col':'Military',
            'Don':'Noble',
            'Dona':'Noble',
            'Dr':'Dr',
            'Jonkheer':'Noble',
            'Lady':'Noble',
            'Major':'Military',
            'Master':'Common',
            'Miss':'Common',
            'Mlle':'Common',
            'Mme':'Common',
            'Mr':'Common',
            'Mrs':'Common',
            'Ms':'Common',
            'Rev':'Clergy',
            'Sir':'Noble',
            'the Countess':'Noble',
            }

def df_transformation(df):
    # lower column name
    logger.info('lower column name')
    df = df.toDF(*[x.lower() for x in df.columns])
    # family member = Siblings/Spouses Aboard + Parents/Children Aboard
    logger.info('family member = Siblings/Spouses Aboard + Parents/Children Aboard')
    df = df.withColumn('family_member', df.sibsp + df.parch)
    # extract tile from name
    logger.info('extract tile from name')
    df = df.withColumn('title', F.split(df.name, ',|\\.')[1])
    df = df.withColumn('title', F.trim(df.title))
    # derive social status from title
    logger.info('derive social status from title')
    df = (df
          .withColumn('social_status', df['title'])
          .replace(to_replace = list(status_map.keys()), value = list(status_map.values()), subset = 'social_status')
         )
    # filling missing values for age
    logger.info('filling missing values for age')
    age_to_fill = int(df.select(F.mean(df.age).alias('mean_age')).collect()[0]['mean_age'])
    df = df.fillna(age_to_fill, subset = 'age')
    # filling missing values for embarked
    logger.info('filling missing values for age')
    df = df.fillna('S', subset = 'embarked')
    
    logger.info('gen last updated column')
    df = df.withColumn('last_updated', F.lit(run_at))
    
    logger.info('Successful Data Transformation')
    return df

df_transformed = df_transformation(df_data)
print(df_transformed.show(10, truncate = False))

############ Data Loading ##################
logger.info('Data Loading')

dy_output = DynamicFrame.fromDF(df_transformed, glueContext, 'data_transformation')

write_to_s3 = glueContext.write_dynamic_frame_from_options(
    frame = dy_output,
    connection_type = 's3',
    connection_options = {
        'path': s3_output_dir,
    },
    transformation_ctx="write_to_s3",
    format = 'parquet'
)

job.commit()
########## importing python lib ##############
import logging
from datetime import datetime, timedelta
from pathlib import pathlib
from pyspark.sql.functions import *

# configing logger
logging.basicConfig(
    format='%(asctime)s :: %(name)s :: %(funcName)s :: Line %(lineno)d :: %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    level = logging.INFO
    )
logger = logging.getLogger(__name__)

############ importing glue ##################
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job


# initilize pyspark cluster conntection
gluecontext = GlueContext(SparkContext.getOrCreate())

# aws store config data
s3_input_dir = '...'
catalog_database = '...'
catalog_table_name = '...'

s3_output_dir = '...'

# read data in term of dynamic frame
# df_ DataFrame object, versatile- can be used elsewhere
# dy_ DynamicFrame object, easy to use but only applied on Glue

# reading data from catalog
dy_data = gluecontext.create_dynamic_frame.from_catalog(
    database = catalog_database,
    table_name = catalog_table_name
)

# reading data from s3 csv
dy_data = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar": '"', "withHeader": True, "separator": ","},
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": s3_input_dir,
        "recurse": True,
    },
)

############# transformation part ###############
def data_transformation(
    dy_data: DynamicFrame
): -> DynamicFrame
    df_transform = dy_data.toDF()
    # code continue
    # ...
    return DynamicFrame.fromDF(df_transform, gluecontext, 'data_transformation')
dy_output = data_transformation(dy_data)


############# Write out in Parquet ################
partitionKeys = []

gluecontext.write_dynamic_frame.from_options(
    frame = dy_output,
    connection_type = 's3',
    conntection_options = {
        'path': s3_output_dir,
        'partitionKeys': partitionKeys
    },
    format = 'parquet'
)
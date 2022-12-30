####################################
#  Calling an API in Python
####################################

import requests
import config as C
import json
from pathlib import Path
from datetime import datetime

def get_weather():

    # My API key is defined in my config.py file
    parameters = {'q'L 'Brooklyn, USA', 'appid': C.API_Key}
    url = 'api.openweathermap.org/data/2.5/weather?'
    result = requests.get(url, parameters)

    # If the api call was successful, get the json and dump it to a file with
    # today's date as the title
    if result.status_code == 200:

        # get the json data
        json_data = result.json()
        file_name = str(datetime.now().date()) + '.json'
        tot_name = Path.cwd() / 'data' / file_name

        with open(tot_name, 'w') as outputfile:

            json.dump(json_data, outputfile)
    else:
        print('Error in API Call')

###################################
#  Setting up PostgreSQL Database
#####################################

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

def make_database():

    dbname = 'WeatherDB'
    username = 'namnh97'
    tablename = 'weather_table'

    engine = create_engine(f'postgresql+psycopg2://{username}@localhost/{dbname')

    if not database_exists(engine.url):
        create_databse(engine.url)

    conn = psycopg2.connect(database = dbname, user = username)

    curr = conn.cursor()

    create_table = f'''
                CREATE TABLE IF NOT EXISTS {tablename}
                (
                    CITY TEXT,
                    COUNTRY TEXT,
                    LAT REAL,
                    LONG REAL,
                    ETL_DATE DATE,
                    HUMIDITY REAL,
                    PRESSURE REAL,
                    MIN_TEMP REAL,
                    MAX_TEMP REAL,
                    TEMP REAL,
                    WEAHTER TEXT
                ) 
                '''

    cur.execute(create_table)
    conn.commit()
    conn.close()

###################################
#  Simple Airflow ETL Task
#####################################

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators import PythonOperator
from pathlib import Path
from airflow.hooks import PostgresHook
import json
import numpy as np

def load_data(ds, **kwargs):

    pg_hook = PostgresHook(postgres_conn_id = 'weather_id')

    file_name = str(datetime.now().date()) + '.json'
    tot_name = Path.cwd() / 'data' / file_name

    with open(tot_name, 'r') as input_file:
        doc = json.load(input_file)

    city = str(doc['name'])
	country = str(doc['sys']['country'])
	lat = float(doc['coord']['lat'])
	long = float(doc['coord']['lon'])
	humid = float(doc['main']['humidity'])
	press = float(doc['main']['pressure'])
	min_temp = float(doc['main']['temp_min']) - 273.15
	max_temp = float(doc['main']['temp_max']) - 273.15
	temp = float(doc['main']['temp']) - 273.15
	weather = str(doc['weather'][0]['description'])
	etl_date = datetime.now().date()

    # check for nan's in the numeric values and then enter into the database
    valid_date = True
    for valid in np.isnan([lat, lon, humid, press, min_temp, max_temp, temp]):
        if valid is False:
            valid_data = False

    insert_cmd = ''' 
    INSERT INTO WEATHER_TABLE
    (
        CITY, COUNTRY, LAT, LONG,
        ETL_DATE, HUMIDITY, PRESSURE,
        MIN_TEMP, MAX_TEMP, TEMP, WEATHER
    )
    VALUES
    (
        city, country, lat, long, humid, press, 
        min_temp, max_temp, temp, weather
    )
    '''

    if valid_data is True:
        pg_hook.run(inser_cmd)

if __name__ == '__main__':
    # get_weather()

    make_database()

    # airflow tasks
    default_args = {
        'owner': 'Nam',
        'depends_on_past': False,
        'email': ['gsn1997@gmail.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 5,
        'retry_delay': timedelta(minutes = 1)
    }

    dag = DAG(
        dag_id = 'weatherDag',
        default_args = default_args,
        start_date = datetime(2022,1,10)
        schedule_interval = timedelta(minutes = 1440)
    )
    
    task1 = BashOperator(
        task_id = 'get_weather',
        bash_command = 'python ~/airflow/dags/src/getWeather.py',
        dag = dag
    )

    task2 = PythonOperator(
        task_id = 'transform_load',
        provide_context = True,
        python_callable = load_data,
        dag = dag
    )

    task1 >> task2
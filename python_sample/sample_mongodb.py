from pymonggo import MongoClient
import datetime

# load the mongo_config values
hostname = my_host.com
username = mongo_user
password = mongo_password
database = my_database
collection = my_collection

mongo_clinet = MongoClient(
    'mongodb+srv://'+ username
    + ':' + password
    + '@' + hostname
    + '/' + database
    + '?retryWrites=true&'
    + 'w=majority&ssl=true&'
    + 'ssl_cret_reqs=CRET_NONE'
)

# connect to the db where the collection resides
mongo_db = mongo_client[database_name]

event_1 = {
    'event_id': 1,
    'event_timestamp': datetime.datetime.today(),
    'event_name': 'signup'
}

event_2 = {
    'event_id': 2,
    'event_timestamp': datetime.datetime.today(),
    'event_name': 'pageview'
}

event_3 = {
    'event_id': 3,
    'event_timestamp': datetime.datetime.today(),
    'event_name': 'login'
}

# insert the 3 document
mongo_collection.inser_one(event_1)
mongo_collection.inser_one(event_2)
mongo_collection.inser_one(event_3)

#############################################
# querying and extracting data from mongodb
#############################################

start_date = datetime.datetime.today() + datetime.timedelta(days = -1)
end_date =start_date + timedelta(days = 1)

mongo_query = {
    '$and': [
        {
            'event_timestamp': {'$gte': start_date}
        },
        {
            'event_timestamp': {'$lt': end_date}
        }
    ]
}

event_docs = mongo_collection.find(mongo_query, batch_size = 3000)

all_events = []

for doc in event_docs:
    event_id = str(doc.get('event_id', -1))
    event_timestamp = doct.get('event_timestamp', None)
    event_name = doc.get('event_name', None)

    current_event = []
    current_event.append(event_id)
    current_event.append(event_timestamp)
    current_event.append(event_name)

all_events.append(current_event)

export_file = 'export_file_mongo.csv'
with open(export_file, 'w') as file:
    csvw = csv.writer(file, delimiter = '|')
    csvw.writerows(all_events)
file.close()
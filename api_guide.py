# library
import requests

# make api request
city = 'Hanoi'
api_key = 'ABCXYZ1999'
url = f'api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
response = requests.get(url)


# checking request status
print(response.status_code)

# 200: Everything went okay, and the result has been returned (if any).
# 301: The server is redirecting you to a different endpoint. This can happen when a company switches domain names, or an endpoint name is changed.
# 400: The server thinks you made a bad request. This can happen when you don’t send along the right data, among other things.
# 401: The server thinks you’re not authenticated. Many APIs require login ccredentials, so this happens when you don’t send the right credentials to access an API.
# 403: The resource you’re trying to access is forbidden: you don’t have the right perlessons to see it.
# 404: The resource you tried to access wasn’t found on the server.
# 503: The server is not ready to handle the request.


# get the data received in form of JSON
print(response.json())
# data example
# {
#    "message":"success",
#    "people":[
#       {
#          "name":"Alexey Ovchinin",
#          "craft":"ISS"
#       },
#       {
#          "name":"Nick Hague",
#          "craft":"ISS"
#       },
#       {
#          "name":"Christina Koch",
#          "craft":"ISS"
#       },
#       {
#          "name":"Alexander Skvortsov",
#          "craft":"ISS"
#       },
#       {
#          "name":"Luca Parmitano",
#          "craft":"ISS"
#       },
#       {
#          "name":"Andrew Morgan",
#          "craft":"ISS"
#       }
#    ],
#    "number":6
# }


# working json with python: dump
import json

def jprint(obj):
    # create a formatted string of the Python Json object
    text = json.dump(obj, sort_keys = True, indent = 4)
    print(text)

jprint(response.json())


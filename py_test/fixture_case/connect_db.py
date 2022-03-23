import json

class StudentDB:
    def __init__(self):
        self.__data = None

    def connect(self, data_files):
        with open(data_files) as json_file:
            self.__data = json.load(json_file)

    def get_data(self, name):
        for student in self.__data['student']:
            if student['name'] == name:
                return student

    def close(self):
        pass
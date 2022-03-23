from connect_db import *
import pytest

##### without the fixture decorator
# db = None
# def setup_module(module):
#     '''
#     Initialize setup/config before the test
#     '''
#     print('----------Set Up------------')
#     global db
#     db = StudentDB()
#     db.connect('data.json')

# def teardown_module(module):
#     '''
#     What to do after the tests are done
#     '''
#     print('----------Tear Down------------')
#     db.close()

@pytest.fixture(scope = 'module')
def db():
    print('----------Set Up------------')
    db = StudentDB()
    db.connect('data.json')
    yield db
    print('----------Tear Down------------')
    db.close()


def test_scott_data(db):
    scott_data = db.get_data('Scott')
    assert scott_data['id'] == 1
    assert scott_data['name'] == 'Scott'
    assert scott_data['result'] == 'pass'

def test_mark_data(db):
    mark_data = db.get_data('Mark')
    assert mark_data['id'] == 2
    assert mark_data['name'] == 'Mark'
    assert mark_data['result'] == 'fail'
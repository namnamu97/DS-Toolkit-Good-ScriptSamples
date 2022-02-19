# logger
import logging
logging.basicConfig(
    format='%(asctime)s :: %(name)s :: %(funcName)s :: Line %(lineno)d :: %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    level = logging.INFO
    )
logger = logging.getLogger(__name__)

# create virtualenv for python 3
$ python3 -m venv python3_venv
$ source python3_venv/bin/activate

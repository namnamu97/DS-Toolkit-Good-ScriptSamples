# In here we build an image that have python 3.9 with pandas and numpy

FROM python:3.9

RUN 
    pip install pandas \
    pip install numpy

WORKDIR /app
COPY first_script_import_pandas.py first_script_import_pandas.py

CMD app/first_script_import_pandas.py

# $ docker build -t test:pandas
# $ docker run test:pandas

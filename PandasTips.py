##############################################################
#Pandas show some obs in numberic column that have wrong format
##############################################################
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df[~df['total_sqft'].apply(is_float)].head()

#############################################################
#interesting functions for pandas:
#############################################################
def convert_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return np.mean(float(tokens[0]), float(tokens[1]))
    try:
        return float(x)
    except:
        return None

##############################################################
#numpy selects to create a new var bases on multiple conditions
##############################################################
col = 'consumption'
conditions = [df[col] >= 400,
              (df[col] < 400) & (df[col] > 200),
              df[col] <= 200
              .
              ]
choices = ['high', 'medium', 'low']

df['class'] = np.select(conditions, choices, default = np.nan)

#if there is only one condition
np.where(condition, value if true, else value)


##############################################################
#Pandas groupby, flexiible returns
##############################################################
df.groupby('A').agg(
    {'B': ['min','max'], 'C': 'sum'}
)

##############################################################
#use of pandas like case when in SQL
##############################################################
df['location'] = df['location'].apply(
    lambda x: 'other' if x in location_stats_less_than_10
    else x
)

##############################################################
#Panas remove outlier of a house price base on mean and stf of location
##############################################################

def remove_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'): #show each loop show entire dataset base on location
        m = np.mean(subdf['price'])
        st = np.std(subdf['price'])
        reduced_df = subdf
        df_out = pd.concat(df_out, reduced_df)

    return df_out

df1 = remove_outlier(df)

##############################################################
#Scatter plot compares the prices of 2 beds and 3 beds apartments
#in a given location
##############################################################

def scatter(df, location):
    bed2 = df[(df['location'] == location) & (df['bed'] == 2)]
    bed3 = df[(df['location'] == location) & (df['bed'] == 3)]

    plt.scatter(bed2['total_sqft'], bed2['price'], color = 'blue', label = '2 beds', s=50)
    plt.scatter(bed3['total_sqft'], bed3['price'], color='green', marker = '+', label='3 beds', s=50)

    plt.xlabel('Total Square Ft Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()

##############################################################
#Save a trained model
##############################################################
import pickle
with open('my_model.pickle', 'wb') as f:
    pickle.dump(model, f)

#save trained data column
import json
columns = {'data_columns': [col.lower() for col in X.columns]}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))

__datacolumn = [col for col in X.columns]
#predict using saved model
def get_esitmated([datacol]):
    try:
        location_index = __datacolumn.index(location.lower())
    except:
        location_index = -1

    x = np.zero(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location_index > = 0:
        x[location_index] = 1

##############################################################
#Model training using Gridsearch CV
##############################################################

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)



##############################################################
#Pandas equivalent statement to SQL LIKE
##############################################################

#To find all the values from the series that starts with a pattern "s":
SQL - WHERE column_name LIKE 's%'
Python - column_name.str.startswith('s')

#To find all the values from the series that ends with a pattern "s":
SQL - WHERE column_name LIKE '%s'
Python - column_name.str.endswith('s')

#To find all the values from the series that contains pattern "s":
SQL - WHERE column_name LIKE '%s%'
Python - column_name.str.contains('s')



##############################################################
#Multiple Dataframes into Excel
##############################################################

import pandas as pd


# Create some Pandas dataframes from some data.
df1 = pd.DataFrame({'Data': [11, 12, 13, 14]})
df2 = pd.DataFrame({'Data': [21, 22, 23, 24]})
df3 = pd.DataFrame({'Data': [31, 32, 33, 34]})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_multiple.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df1.to_excel(writer, sheet_name='Sheet1')
df2.to_excel(writer, sheet_name='Sheet2')
df3.to_excel(writer, sheet_name='Sheet3')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

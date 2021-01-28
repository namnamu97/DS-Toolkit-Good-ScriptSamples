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

#break numerical values into specified bins
cut_labels_4 = ['silver', 'gold', 'platinum', 'diamond']
cut_bins = [0, 70000, 100000, 130000, 200000]
df['cut_ex1'] = pd.cut(df['ext price'], bins=cut_bins, labels=cut_labels_4)


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


##############################################################
#Import Bigquery to Pandas
##############################################################

sql = """
    SELECT *, 'train' AS train_test
    FROM `TestDemo.1mgtest_train`
    UNION ALL
    SELECT *, 'test' AS train_test
    FROM `TestDemo.1mgtest_test`
    """

project_id = 'seismic-hexagon-295906'

df = pd.read_gbq(sql, project_id = project_id, dialect = 'standard')

df.head()

##############################################################
#Statistical Testing for Feature Selection
##############################################################

#anova testing for numerical input---------------------
from sklearn.feature_selection import f_classif

p_vals = f_classif(df.select_dtypes(['float64','int64']), df['churn'])[1]
cols = df.select_dtypes(['float64','int64']).columns
                   
for col, p_val in zip(cols, p_vals):
    if p_val > 0.05:
        print(f'anova test for {col} is insignificant')
        
#chi2 testing for categorical-------------------------
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

#chi2 in sklearn require input to be numerical
df_cat = df.select_dtypes(['object', 'bool']).drop(['churn', 'train_test'], axis = 1).apply(LabelEncoder().fit_transform)

p_vals = chi2(df_cat, df['churn'])[1]
cols = df_cat.columns

for col, p_val in zip(cols, p_vals):
    if p_val > 0.05:
        print(f'anova test for {col} is insignificant')

##############################################################
#Calculating MAP for RecSys
##############################################################
        
import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# calculate the MAP
    actual = []
    pred = []
    for k,_ in recomendations_dict.items():
        actual.append(list(interactions_valid_dict[k]))
        pred.append(list(recomendations_dict[k]))

    result = mapk(actual,pred)

    
##############################################################
#PVC-2D Viz clustering
##############################################################

from sklearn.decomposition import PCA

def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    
    return df_matrix

plt.figure(figsize = (10,12))
sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, 
                palette="Set2")
plt.show()

##############################################################
# Elbow Viz for KMeans CLustering
##############################################################

# elbow method

#model fit: helper function
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer

# Use the quick method and immediately show the figure
kelbow_visualizer(KMeans(random_state=50), cluster_stand, k=(2,26))

##############################################################
# Viz for Clustering Attributes
##############################################################

# model fitting

from sklearn.cluster import KMeans

cluster_info = cluster_stand.copy()

kmeans_stand = KMeans(n_clusters = 6, random_state = 50)
kmeans_stand.fit(cluster_info)

# Gen df that contains information in each cluster

cluster_info['cluster'] = kmeans_stand.predict(cluster_info)

cluster_info = cluster_raw.reset_index().merge(cluster_info.reset_index()[['user_id', 'cluster']], on = 'user_id')

# define aggregations for the groupby
agg = {'user_id':'count'}
for col in cluster_info.columns[1:]:
    agg[col] = 'mean'
    
# cluster info table
cluster_info = cluster_info.groupby('cluster').agg(agg).\
                                rename(columns = {'user_id': 'count'}).transpose().round(4)

# unstack the cluster attributes
tidy = cluster_info.transpose().drop(['count', 'cluster', 'num_trans'], axis = 1).reset_index()
tidy = tidy.melt(id_vars = 'cluster')

plt.figure(figsize = (20,16))
sns.set_style('whitegrid')
ax = sns.barplot(x = 'cluster', y = 'value', hue = 'variable', palette = 'Set2',data = tidy)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='25') # for legend title
plt.title('Voucher Clustering', fontsize = 20)
plt.tight_layout()
plt.show()


##############################################################
# Custom Transformer
##############################################################
#Custom transformer we wrote to engineer features ( bathrooms per bedroom and/or how old the house is in 2019  ) 
#passed as boolen arguements to its constructor
class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, bath_per_bed = True, years_old = True ):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        #Check if needed 
        if self._bath_per_bed:
            #create new column
            X.loc[:,'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            #drop redundant column
            X.drop('bathrooms', axis = 1 )
        #Check if needed     
        if self._years_old:
            #create new column
            X.loc[:,'years_old'] =  2019 - X['yr_built']
            #drop redundant column 
            X.drop('yr_built', axis = 1)
            
        #Converting any infinity values in the dataset to Nan
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        #returns a numpy array
        return X.values

##############################################################
# Pipeline Creation
##############################################################
set_config(display = 'diagram')

numerical_col = X.select_dtypes(['int64', 'float64']).columns
categorical_col = X.select_dtypes(['object', 'bool']).columns

transformation = [('cat', OneHotEncoder(). categorical_col), ('num', MinMaxScaler(), numerical_col)]

col_transformer = ColumnTransformer(transformer = transformation)

model = SVR()

pipline = Pipeline(steps = [
    ('prep', col_transform),
    ('model', model)
    ]
              
pipeline #show the pipeline

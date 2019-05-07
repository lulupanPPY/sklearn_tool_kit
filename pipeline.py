from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.externals import joblib
import sklearn
from sklearn.impute import SimpleImputer
import numpy as np

df = pd.read_csv('D://Kaggle//Titanic//train.csv', header=0,iterator=False)
df_test = pd.read_csv('D://Kaggle//Titanic//test.csv', header=0,iterator=False)

df.drop(columns=['PassengerId'], inplace=True)
df.drop(columns=['Name'], inplace=True)
df.drop(columns=['Ticket'], inplace=True)
#df.drop(columns=['Survived'], inplace=True)



cols = df.columns
x_cols = df.columns.drop(['Survived'],errors='raise')
y_col = 'Survived'

df_X = df[x_cols]
df_Y = df[y_col]

dtypes = df_X.dtypes


enum_cols = []
num_cols = []
for i in range(0,len(x_cols)):
    if dtypes[i] == 'object':
        enum_cols.append(x_cols[i])
    else:
        num_cols.append(x_cols[i])

imp_num_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_enum = SimpleImputer(missing_values=np.nan,strategy="constant",fill_value='unkown')

imp_enum = imp_enum.fit(df[enum_cols])
imp_num = imp_num_median.fit(df[num_cols])

'''
#impute
print('show me the results:')
df_enum = pd.DataFrame(imp_enum.transform(df[enum_cols]),columns=enum_cols)
df_num = pd.DataFrame(imp_num.transform(df[num_cols]),columns=num_cols)
print (df_enum.shape)
print (df_num.shape)
df = pd.DataFrame.merge(df_enum,df_num,left_index=True, right_index=True)
'''



mapper = DataFrameMapper(
    [([d], imp_num ) for d in num_cols]+
    [([d], [imp_enum,sklearn.preprocessing.LabelBinarizer()]) for d in enum_cols]
    #['Age',]
#    ('Sex', sklearn.preprocessing.LabelBinarizer()),
    #(['Age'], sklearn.preprocessing.KBinsDiscretizer(n_bins=10, encode='onehot')),
#    (['Age'], sklearn.preprocessing.StandardScaler())
)

CLF_L1_LR=LogisticRegression(C=0.1, penalty='l1', max_iter=2000)

#pipeline
pipeline = Pipeline([('mapper',mapper),('lr',CLF_L1_LR)])
pipeline.fit(df[x_cols],df[y_col])
#print (pipeline['lr'].summary())

y_hat = pd.DataFrame(pipeline.predict_proba(df[x_cols]))
df_out = pd.DataFrame.merge( y_hat, pd.DataFrame(df[y_col]),left_index=True, right_index=True)
df_out.to_csv('D://Kaggle//Titanic//test_M2_out.csv')
'''
df=pd.DataFrame(mapper.fit_transform(df),columns=mapper.transformed_names_)
#df = pd.DataFrame.merge(df_enum,df_num,left_index=True, right_index=True)

#predict
df_test = pd.read_csv('D://Kaggle//Titanic//test_M1.csv', header=0,iterator=False)
df_PassengerID = pd.DataFrame(df_test['PassengerId'])
df_test.drop(columns=['PassengerId'], inplace=True)
df_test.drop(columns=['Name'], inplace=True)
df_test.drop(columns=['Ticket'], inplace=True)


df_test=pd.DataFrame(mapper.transform(df_test),columns=mapper.transformed_names_)
df_test.to_csv('D://Kaggle//Titanic//test_M1_out.csv')
print (df.columns)
print (df_test.columns)
'''



from sklearn_pandas import DataFrameMapper, cross_val_score
import pandas as pd
import numpy as np
import sklearn.preprocessing, sklearn.decomposition,sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                    'children': [4., 6, 3, 3, 2, 3, 5, 4],
                    'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})

mapper = DataFrameMapper([
        ('pet', sklearn.preprocessing.LabelBinarizer()),
        (['children'], sklearn.preprocessing.StandardScaler())
])

data_copy=mapper.fit_transform(data.copy())

print (data_copy)
from tools import *
from tools import AUC_plot
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


###########################################################
#           preprocessing
###########################################################
##############################
#        reading data
##############################
#D:\Kaggle\BlackFriday
#df1 = dataframe('D://Kaggle//BlackFriday//BlackFriday.csv',0)
df1 = dataframe('D://Kaggle//stock//stock001.csv',0)
df2 = dataframe('D://Kaggle//stock//test001.csv',0)
##############################
#        show if there are missing values
##############################
print(df1.get_df_types())


##############################
#        drop unwanted features
##############################
df1.drop('code')
df1.drop('date')
df2.drop('code')
df2.drop('date')
print(df1.get_df_types())
#df1.drop('PassengerId')
#df1.drop('Ticket')
#df1.drop('Cabin')
#df1.drop('User_ID')
#df1.drop('Product_ID')


##############################
#      make the data visible
##############################

#df1.get_hist_num()

df1.xy_corr_num_dense('rise_in_7','buy_point')
#df1.xy_corr_num_sparse('Pclass','Survived')
#df1.xy_corr_num_sparse('Sex','Pclass','Survived')
#print (df1.df.groupby('Stay_In_Current_City_Years').count())


#df1.show_enum_target_corr('Survived')
#df1.show_num_target_corr_single('Survived','Fare',10)


##############################
#        fill missing values
##############################
#df1.fill_miss_num('Age','avg')
#df1.fill_miss_enum('Embarked','unknown')
#df1.fill_miss_num('Age','median')


###############################
#      some feature engineering
###############################
#df1.discretize_single('Age',10)
#df1.binarize_all()


# Machine Learning Algorithm (MLA) Selection and Initialization

######################################
#            Fitting
######################################
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    #XGBClassifier()
]

train_x, test_x, train_y, test_y = df1.train_test_split('buy_point',888)
CLF_LR = LogisticRegression(penalty='l1',C=1, tol=0.001,max_iter=2000, solver='saga')
CLF_LR.fit(train_x,train_y)
coef = CLF_LR.coef_
print (coef)

for i in range(len(coef[0])):
    print (df1.get_col_names()[i],coef[0,i])

print(CLF_LR.intercept_)


Y_pred_train = CLF_LR.predict_proba(train_x)[:,1]
AUC_plot(Y_pred_train,train_y,'training')

print (CLF_LR.predict(df2.df))

'''
cv_results = model_selection.cross_validate(CLF_LR, df1.df[train_x.columns], df1.df['Survived'], cv  = 5)
print (cv_results)

Y_pred_test = CLF_LR.predict_proba(test_x)[:,1]
AUC_plot(Y_pred_test,test_y,'testing')
'''




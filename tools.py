from sklearn import preprocessing
import pandas as pd
import sys
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
import numpy as np
from collections import Counter

chk_size = 100
class dataframe:
    def __init__(self,file_path,header):
        reader = pd.read_csv(file_path, header=header,iterator=True)
        loop = True
        chunkSize = 100
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")

        self.df = pd.concat(chunks, ignore_index=True)



    def read_csv (file_path,header):
        df = pd.read_csv(file_path,header=header,chunksize=chk_size)
        return df
# get a list of col names
    def get_col_names(self):
        return self.df.keys()
# get a brief view about the data
    def get_brief_view(self):
        return self.df.info()
# get a list of column data types
    def get_df_types(self):
        return self.df.dtypes
    def get_df_sample(self,n):
        return self.df.sample(n)

# print min,max,min,variance for numeric features;
    def get_num_statistics(self):
        return self.df.describe()

# print histogram for single enumerate feature
    def plot_enum_hist(self,col):
        plt.figure(figsize=(12, 10))
        sns.set_style("darkgrid")
        sns.countplot(self.df[col])
        print(self.df[col])
        plt.show()
        return

# print histogram for enumerate features
    def plot_enum_hist_all(self):
        enum_list = self.get_enum_feature_list()
        if len(enum_list) <= 4:
            row_size = 2
            col_size = 2
        elif len(enum_list) <= 16:
            row_size = 4
            col_size = 4
        elif len(enum_list) <= 30:
            row_size = 5
            col_size = 6
        else:
            row_size = 10
            col_size = 10

        fig, axes = plt.subplots(row_size,col_size)
        sns.set_style("darkgrid")

        idx = 0
        for feature in enum_list:
            print (feature)
            sns.countplot(self.df[feature], ax=axes[idx % row_size, idx // row_size])
            idx +=1

        plt.tight_layout()
        plt.show()
        return

    def apply_log(self,col):
        self.df[col] = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)

    def detect_outliers(self,df, n, features):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        outlier_indices = []

        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR
            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        return multiple_outliers
# get enum features
    def get_enum_feature_list(self):
        #print (self.get_col_names())
        cols = self.get_df_types().index
        idx = 0
        enum_list=[]
        for type in self.get_df_types():
            if (type == 'object'):
                enum_list.append(cols[idx])
            idx+=1
        return enum_list
# print null info
    def get_null_info(self):
        return self.df.isnull().sum()
# print histogram
    def get_hist_num(self,bins=10):
        self.df.hist(bins=bins)
        plt.show()
        return




# todo
# Standardize all numeric columns,
    def Standardize_all_numeric(self):
        return

# Standardize single numeric column
    def Standardize_single_col(col):
        return

# Discretize all numeric features
    def discretize_all(self):
        return

#Discretize single numeric features
    def discretize_single(self,col,k):
        enc = KBinsDiscretizer(n_bins=k, encode='onehot')
        X = self.df[col].values.reshape(-1, 1)
        X_binned = enc.fit_transform(X)
        tmp = pd.DataFrame(X_binned.toarray())
        columns = []
        for i in range(0, k):
            columns.append(col + '_' + str(i))
        tmp.columns = columns
        self.df.drop(columns=[col], inplace=True)
        df = pd.DataFrame.merge(self.df, tmp, left_index=True, right_index=True)

        return

# binarize all enum features
    def binarize_all(self):
        for col in self.get_enum_feature_list():
            self.binarize_single(col)
        return

    def drop (self,col):
        self.df.drop(columns=[col], inplace=True)
        return

# binarize single enum features
    def binarize_single(self,col):
        tmp = pd.get_dummies(self.df[col])
        new_colname = []
        for col_1 in tmp.columns:
            new_colname.append(col + '_' + col_1)
        tmp.columns = new_colname
        self.df.drop(columns=[col], inplace=True)
        # print ('dropping column '+col)
        df = pd.DataFrame.merge(self.df, tmp, left_index=True, right_index=True)

        return

#split into train, test, validate
    def split(train,test,validate):
        return
# split data into training and testing
    def train_test_split (self,target,rand):
        X_names = self.df.columns.tolist()
        X_names.remove(target)
        train_x, test_x, train_y, test_y = model_selection.train_test_split(self.df[X_names], self.df[target],
                                                                                random_state=rand)
        return train_x, test_x, train_y, test_y
# fill missing values
# for numeric values , choose median or avg
    def fill_miss_num(self,col,fun):
        if self.df[col].dtype == 'object':
            print('object data type encountered!')
            sys.exit(1)
        if fun == 'median':
            self.df[col].fillna(self.df[col].median(), inplace=True)
        elif fun == 'avg':
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        else:
            print('Error :wrong arguments:')
            sys.exit(1)


        return

    def fill_miss_enum(self,col,val):
        if self.df[col].dtype != 'object':
            print('not object data type encountered!')
            sys.exit(1)

        self.df[col].fillna(val, inplace=True)


        return

# print correlations:
    def get_corr(self):
        #print (self.df.corr())
        return self.df.corr()

#show enum feature correlations
    def show_enum_target_corr(self,Y_name):
        for x in self.df.columns:
            if self.df[x].dtype != 'float64' and x != Y_name:
                print('Target Correlation by:', x)
                print(self.df[[x, Y_name]].groupby(x, as_index=False).mean())
                print('-' * 10, '\n')


#print correlation heat map
    def correlation_heatmap(self):
        _, ax = plt.subplots(figsize=(14, 12))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        _ = sns.heatmap(
            self.df.corr(),
            cmap=colormap,
            square=True,
            cbar_kws={'shrink': .9},
            ax=ax,
            annot=True,
            linewidths=0.1, vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 6}
        )

        plt.title('Pearson Correlation of Features', y=1.05, size=15)

    def xy_corr_num_sparse(self,x,y):
        g = sns.factorplot(x=x, y=y, data=self.df, kind="bar", size=6,
                           palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")
        plt.show()

    def xy_corr_num_sparse(self,x1,x2,y):
        g = sns.factorplot(x=x1, y=y, hue =x2 ,data=self.df, kind="bar", size=6,
                           palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")
        plt.show()

    def num_boxplt(self,col):
        df = self.df[col]
        df = df.dropna(how='any',axis=0)
        plt.figure(figsize=[16, 12])
        plt.subplot(231)
        plt.boxplot(x=df, showmeans=True, meanline=True)
        plt.title(col + ' Boxplot')
        plt.ylabel(col)
        plt.show()

    def xy_corr_num_dense(self,x,y):
        g = sns.FacetGrid(self.df, col=y)
        g = g.map(sns.distplot, x)
        plt.show()

# show correlation of target and numeric features after discretized
    def show_num_target_corr_single(self,Y_name,col,k=10):
        tmp = discretize(self.df, col, k)
        print(type(self.df[Y_name]))
        tmp = pd.DataFrame.merge(tmp, pd.DataFrame(self.df[Y_name]), left_index=True, right_index=True)

        for x in tmp.columns:
            if x != Y_name:
                print(tmp[[x, Y_name]].groupby(x, as_index=False).mean())

        return



def AUC_plot(Y_hat,Y,title):
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(Y, Y_hat, pos_label=1)
    roc_auc = auc(fpr_train, tpr_train)
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    return

def show_explicit_CORR(df,thres):
    names = df.keys()
    for index, row in df.iterrows():
        for i in range (0,len(row)):
            if row[i] > thres and index != names[i]:
                print (index,' and ', names[i],' has a correlation of:\t',row[i])

def discretize(df,col,k):
    enc = KBinsDiscretizer(n_bins=k, encode='onehot')
    X = df[col].values.reshape(-1, 1)
    X_binned = enc.fit_transform(X)
    tmp = pd.DataFrame(X_binned.toarray())
    columns = []
    for i in range(0, k):
        columns.append(col + '_' + str(i))
    tmp.columns = columns
    return  pd.DataFrame(tmp)



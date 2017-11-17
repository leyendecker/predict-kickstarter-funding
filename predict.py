# Import libraries
from datetime import datetime as dt
import json
import numpy as np
import pandas as pd
from scipy import stats
from time import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import preprocessing, cross_validation, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.sparse import hstack
from scipy.sparse.csr import csr_matrix
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV

pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Read data
data = pd.read_csv("data/processed.csv")
data['launched_at'] = [dt.strptime(x, '%Y-%m-%d %H:%M:%S') for x in data['launched_at']]
data['deadline'] = [dt.strptime(x, '%Y-%m-%d %H:%M:%S') for x in data['deadline']]
data['name'] = data['name'].fillna('')
data['desc'] = data['desc'].fillna('')
data['keywords'] = data['keywords'].fillna('')
data['country'] = data['country'].astype('category')


print "Data read successfully!"

models = [
    ('SGD', SGDClassifier(random_state=42)),
    # ('LR', LogisticRegression()),
    # ('NB', GaussianNB()),
    ('DT', DecisionTreeClassifier(random_state=42)),
    ('RF', RandomForestClassifier(random_state=42))
]

kw_features_r = range(0, 200, 50)
name_features_r = range(0, 2501, 500)
desc_features_r = range(0, 2501, 500)

total = len(desc_features_r) * len(name_features_r) * len(desc_features_r)
index = 0
res = []
for desc_features in desc_features_r:
    for name_features in name_features_r:
        for kw_features in kw_features_r:
            X = None
            if kw_features > 0:
                X = hstack((X, TfidfVectorizer(max_features=kw_features).fit_transform(list(data['keywords']))))
            if name_features > 0:
                X = hstack((X, TfidfVectorizer(max_features=name_features).fit_transform(list(data['name']))))
            if desc_features > 0:
                X = hstack((X, TfidfVectorizer(max_features=desc_features).fit_transform(list(data['desc']))))
                
            X = hstack((X, np.column_stack((data.index.tolist(), data['deadline'].apply(lambda x: x.year + x.month/12.0).as_matrix()))))
            X = hstack((X, np.column_stack((data.index.tolist(), data['launched_at'].apply(lambda x: x.year + x.month/12.0).as_matrix()))))
            X = hstack((X, np.column_stack((data.index.tolist(), data['days_open'].tolist()))))
            X = hstack((X, np.column_stack((data.index.tolist(), data['disable_communication'].tolist()))))
            X = hstack((X, np.column_stack((data.index.tolist(), data['goal'].tolist()))))
            X = hstack((X, np.column_stack((data.index.tolist(), data['country'].cat.codes))))

            y = data['final_status']
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X.toarray(), y, test_size=0.05, random_state=42)

            sample = {}
            sample['kw_features']   = kw_features
            sample['name_features'] = name_features
            sample['desc_features'] = desc_features
            for name, model in models:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                sample[name] = score

            res.append(sample)
            pd.DataFrame(res).to_csv('data/results.csv', index=False, sep= ';')
            index += 1
            print "%s/%s - %.2f%%" % (index,total,(index*100.0/total))


    # print "%s: %f (acc), %f (f1), %f (recall) %f (precision)" % (name, acc.mean(), f1.mean(), recall.mean(), precision.mean())

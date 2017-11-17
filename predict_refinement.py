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

param_grid = {
  'n_estimators' : [10, 50, 100, 200, 500, 1000],
  'criterion'    : ['gini', 'entropy']
}

model = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=20, criterion='entropy'),cv=5,
    param_grid=param_grid)

X = None
X = hstack((X, TfidfVectorizer(max_features=100).fit_transform(list(data['keywords']))))
X = hstack((X, TfidfVectorizer(max_features=1500).fit_transform(list(data['name']))))
X = hstack((X, TfidfVectorizer(max_features=2500).fit_transform(list(data['desc']))))
X = hstack((X, np.column_stack((data.index.tolist(), data['deadline'].apply(lambda x: x.year + x.month/12.0).as_matrix()))))
X = hstack((X, np.column_stack((data.index.tolist(), data['launched_at'].apply(lambda x: x.year + x.month/12.0).as_matrix()))))
X = hstack((X, np.column_stack((data.index.tolist(), data['days_open'].tolist()))))
X = hstack((X, np.column_stack((data.index.tolist(), data['disable_communication'].tolist()))))
X = hstack((X, np.column_stack((data.index.tolist(), data['goal'].tolist()))))
X = hstack((X, np.column_stack((data.index.tolist(), data['country'].cat.codes))))

y = data['final_status']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X.toarray(), y, test_size=0.05, random_state=42)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print score



res = pd.DataFrame([{'score':res[0],'std':res[1],'n_estimators':res[2]['n_estimators'],'criterion':res[2]['criterion']} for res in zip(model.cv_results_['mean_test_score'],model.cv_results_['std_test_score'],model.cv_results_['params'])])
res.to_csv('data/results_refinement.csv', index=False, sep= ';')

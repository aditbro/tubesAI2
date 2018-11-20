import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.model_selection import GridSearchCV
import pickle

data = pd.read_csv("tubes2_HeartDisease_train.csv").replace('?',np.NaN)
shuffle(data.values)
data = data.drop("Column12",axis=1)
data = data.drop("Column13",axis=1)
i = 0

for count in data.count(axis=1):
    if(count <= 7):
        data = data.drop(i, axis = 0)
    i += 1
# print(data.count())
# print(X.iloc[[1]])

# X.append([data.columns[12:13]])
X = data[data.columns[0:11]]
y = data['Column14']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print(X)
# X_test = X_train.copy()
# y_test = y_train.copy()

# removes NaN
imp_modus = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp_modus.fit(X_train.values)
clear_train_NaN = imp_modus.transform(X_train)
clear_test_NaN = imp_modus.transform(X_test)

# scaling features
# X_train_scaled = preprocessing.scale(clear_train_NaN)
# X_test_scaled = preprocessing.scale(clear_test_NaN)
# X_train = X_train_scaled
# X_test = X_test_scaled

X_train = clear_train_NaN
X_test = clear_test_NaN
# x = np.random.random_integers(1, 100, 5)
# print(x)
# x = data[data.columns[4]].values
# plt.hist(x, bins=50)
# plt.ylabel('No of times')
# plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt ='d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

clf = MLPClassifier(max_iter=9000,solver='adam', alpha=0.0001,hidden_layer_sizes=(10,9,6,5,5), random_state=1,)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %0.2f" % (accuracy_score(y_test,y_pred)*100))
# # plot_confusion_matrix(confusion_matrix(y_test, y_pred), clf.classes_, title = "Neural Network Confusion Matrix")

# layer = []
# for i in range (3,7):
#     layer.append([i])
# for i in range (7,12):
#     for j in range(3,7):
#         layer.append([i,j])
# for i in range(7,12):
#     for j in range(7,12):
#         for k in range(3,7):
#             layer.append([i,j,k])
# parameters = {'solver': ['adam'], 'max_iter': [5000], 'hidden_layer_sizes':layer, 'random_state':[0]}
# clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1,cv=10, scoring='accuracy')

# clf.fit(X_train,y_train)
# print(clf.score(X_train, y_train))

# print("Best parameters set found on development set:")
# print(clf.best_params_)
# pickle.dump(clf, open('clf.sav', 'wb'))
# y_pred = clf.predict(X_test)
# print("Accuracy: %0.2f" % (accuracy_score(y_test,y_pred)*100))
gnb = GaussianNB()
naive_model = gnb.fit(X_train, y_train)
# print(naive_model)
y_pred = naive_model.predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
print("Accuracy: %0.2f" % (accuracy_score(y_test,y_pred)*100))
conf_matrix = (confusion_matrix(y_test, y_pred))
plot_confusion_matrix(conf_matrix, naive_model.classes_, title = "Naive Bayes Confusion Matrix")

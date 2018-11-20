print("importing sklearn...")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
print("importing input interpreter...")
from modules import IO
print("importing pickle")
import pickle

model_dir = 'models/'

inp = IO.InputDataInterpreter(filename="data/tubes2_HeartDisease_train.csv")
print(inp.data_stat)
test_data = IO.TestDataInterpreter(filename="data/tubes2_HeartDisease_test.csv", data_stat=inp.data_stat)

train_data = inp.data[0:600]
train_target = inp.target[0:600]

test_data = inp.data[600:]
test_target = inp.target[600:]

train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)

print("Training data...")
clf = MLPClassifier(max_iter=9000, solver='lbfgs', alpha=1e-5 ,hidden_layer_sizes=(11,5), random_state=0, shuffle=True)
clf_model = clf.fit(train_data, train_target)

print("predicting data...")
result = clf_model.predict(test_data)

fit = 0
for i in range(len(result)):
	if (result[i] == test_target[i]):
	    fit += 1

print('MLP fit {}%'.format(fit/len(result) * 100))


dt = DecisionTreeClassifier()
dt_model = dt.fit(train_data, train_target)

result = dt_model.predict(test_data)
fit = 0
for i in range(len(result)):
	if (result[i] == test_target[i]):
	    fit += 1

print('DT fit {}%'.format(fit/len(result) * 100))

nb = GaussianNB()
nb_model = nb.fit(train_data, train_target)

result = nb_model.predict(test_data)
fit = 0
for i in range(len(result)):
    if (result[i] == test_target[i]):
	    fit += 1

print('NB fit {}%'.format(fit/len(result) * 100))

knn = KNeighborsClassifier(15)
knn_model = knn.fit(train_data, train_target)

result = knn_model.predict(test_data)
fit = 0
for i in range(len(result)):
	if (result[i] == test_target[i]):
		fit += 1

print('KNN fit {}%'.format(fit/len(result) * 100))

pickle.dump(clf_model, open(model_dir+'MLP', 'wb'))
pickle.dump(dt_model, open(model_dir+'DT', 'wb'))
pickle.dump(knn_model, open(model_dir+'KNN', 'wb'))
pickle.dump(nb_model, open(model_dir+'NB', 'wb'))
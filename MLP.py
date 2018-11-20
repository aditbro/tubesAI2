print("importing sklearn...")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
print("importing input interpreter...")
from modules import IO
print("importing pickle")
import pickle

model_filename = 'models/MLP'

inp = IO.InputDataInterpreter(filename="data/tubes2_HeartDisease_train.csv")
test_data = IO.TestDataInterpreter(filename="data/tubes2_HeartDisease_test.csv")

kf = KFold(n_splits=5)
kf.get_n_splits(inp.data)

train_data = []
train_target = []

test_data = []
test_target = []
for train_index, test_index in kf.split(inp.data):
	for idx in train_index:
		train_data.append(inp.data[idx])
		train_target.append(inp.target[idx])

	for idx in test_index: 
		test_data.append(inp.data[idx])
		test_target.append(inp.target[idx])

print("Training data...")
clf = MLPClassifier(max_iter=9000, solver='lbfgs', verbose=True, alpha=1e-5 ,hidden_layer_sizes=(13,13,13,13,13,13,13,13,13,13), random_state=0, shuffle=True)
clf.fit(train_data, train_target)
# clf = pickle.load(open(model_filename,'rb'))

print("predicting data...")
result = clf.predict(test_data)

fit = 0
for i in range(len(result)):
    if result[i] == test_target[i]:
        fit += 1

print('MLP fit {}%'.format(fit/len(result) * 100))


dt = DecisionTreeClassifier()
dt_model = dt.fit(train_data, train_target)

result = dt_model.predict(test_data)
fit = 0
for i in range(len(result)):
    if result[i] == test_target[i]:
        fit += 1

print('DT fit {}%'.format(fit/len(result) * 100))

nb = GaussianNB()
nb_model = nb.fit(train_data, train_target)

result = nb_model.predict(test_data)
fit = 0
for i in range(len(result)):
    if result[i] == test_target[i]:
        fit += 1

print('NB fit {}%'.format(fit/len(result) * 100))

knn = KNeighborsClassifier(n_neighbors, weights=weights)
knn_model = knn.fit(train_data, train_target)

result = knn_model.predict(test_data)
fit = 0
for i in range(len(result)):
    if result[i] == test_target[i]:
        fit += 1

print('KNN fit {}%'.format(fit/len(result) * 100))

pickle.dump(clf, open(model_filename, 'wb'))
print("importing sklearn...")
from sklearn.neural_network import MLPClassifier
print("importing input interpreter...")
from modules import IO
print("importing pickle")
import pickle

model_filename = 'models/MLP'

inp = IO.InputDataInterpreter(filename="data/tubes2_HeartDisease_train.csv")
test_data = IO.TestDataInterpreter(filename="data/tubes2_HeartDisease_test.csv")

print(inp.data, inp.target)

print("Training data...")
clf = MLPClassifier(max_iter=9000, solver='lbfgs', verbose=True, alpha=1e-5 ,hidden_layer_sizes=(13,13,13,13,13,13,13,13,13,13), random_state=0, shuffle=True)
clf.fit(inp.data, inp.target)

print("predicting data...")
result = clf.predict(inp.data)

fit = 0
for i in range(len(result)):
    print(result[i], inp.target[i])
    if result[i] == inp.target[i]:
        fit += 1

print('fit {}%'.format(fit/len(result) * 100))

pickle.dump(clf, open(model_filename, 'wb'))
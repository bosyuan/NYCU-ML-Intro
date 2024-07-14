# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    # if len(y) == 0:
    #     return 0
    count0 = 0
    count1 = 0
    for l in y:
        if l == 1: count1 += 1
        else : count0 += 1
    return 1 - (count0/len(y))**2 - (count1/len(y))**2

#This is a function that computes impurity with a given probability p of class m
def entropy_impurity(p):
    if p == 0:
        return 0
    else:
        return (-1) * p * np.log(p)
    
# This function computes the entropy of a label array.
def entropy(y):
    # if len(y) == 0:
    #     return 0
    count0 = 0
    count1 = 0
    for l in y:
        if l == 1: count1 += 1
        else : count0 += 1
    return entropy_impurity(count0 / len(y)) + entropy_impurity(count1 / len(y))

class Node:
    def __init__(self, depth=None):
        self.predict_class = -1
        self.feature = -1
        self.threshhold = 0
        self.left = None
        self.right = None
        self.depth = depth

# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.root = None
        self.feature_importance = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
        
    def generate_tree(self, X, y, depth):
        node = Node(depth)
        count0 = 0
        count1 = 0
        for l in y:
            if l == 1: count1 += 1
            else : count0 += 1
        if (count0 > count1):
            node.predict_class = 0
        else:
            node.predict_class = 1
        
        if depth >= self.max_depth:
            # print("create leaf", depth, node.predict_class)
            return node
        
        bestf, bestw = self.split_attribute(X, y)
        if bestf!= None:
            X_left = []
            y_left = []
            X_right = []
            y_right = []
            for i in range(len(y)):
                if (X[i][bestf] > bestw):
                    X_right.append(X[i])
                    y_right.append(y[i])
                else:
                    X_left.append(X[i])
                    y_left.append(y[i])

            node.feature = bestf
            node.threshhold = bestw
            node.left = self.generate_tree(np.array(X_left), np.array(y_left), depth+1)
            node.right = self.generate_tree(np.array(X_right), np.array(y_right), depth+1)
            # print("node depth: ", node.depth)
            # if (node.left != None): print("left ", node.left.predict_class)
            # if (node.right != None): print("right ", node.right.predict_class)
            self.feature_importance[bestf] += 1
            # print("create node: ", depth, node.predict_class, bestf, bestw, len(y_left), len(y_right))
        # else:
            # print("create leaf", depth, node.predict_class)
        return node

        
    def split_attribute(self, X, y):
        bestf = None
        bestw = None
        if (len(y) <= 1):
            return bestf, bestw
        MinEnt = self.impurity(y)

        for i in range(X.shape[1]):
            lst = []
            for j in range(X.shape[0]):
                lst.append((X[j][i], y[j]))
            sorted_lst = sorted(lst)

            for j in range(len(sorted_lst) - 1):
                w = (sorted_lst[j][0] + sorted_lst[j+1][0]) / 2
                y_left = []
                y_right = []
                for k in range(len(sorted_lst)):
                    if sorted_lst[k][0] > w:
                        y_right.append(sorted_lst[k][1])
                    else:
                        y_left.append(sorted_lst[k][1])
                if (len(y_right) == 0 or len(y_left) == 0): 
                    continue

                Ent = (len(y_right) * self.impurity(y_right) + len(y_left) * self.impurity(y_left)) / len(sorted_lst)
                if (Ent < MinEnt):
                    MinEnt = Ent
                    bestf = i
                    bestw = w
        return bestf, bestw

    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.feature_importance = np.zeros(X.shape[1])
        self.root = self.generate_tree(X, y, 0)
        return
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            node = self.root
            while(node.right != None):
                if (X[i][node.feature] > node.threshhold):
                    node = node.right
                else:
                    node = node.left
            y.append(node.predict_class)
            # print(node.predict_class)
        return np.array(y)

    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        plt.barh(columns, self.feature_importance, color='blue')
        plt.title('Feature Importance')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []
        self.classifiers = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        # Initialize weights
        n_samples = len(X)
        w = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # Create a decision tree with max_depth=1
            tree = DecisionTree(criterion=self.criterion, max_depth=1)

            # Weighted random sample of training data
            X_rand = []
            y_rand = []
            weight_sum = []
            weight_sum.append(0)
            for j in range(1, len(w)):
                weight_sum.append(weight_sum[j-1] + w[j-1])
            if (i != 0):
                for j in range(X.shape[0]):
                    random_float = np.random.uniform(0, 1)
                    sample = len(weight_sum) - 1
                    for k in range(len(weight_sum) - 1):
                        if weight_sum[k] <= random_float < weight_sum[k + 1]:
                            sample = k
                            break
                    X_rand.append(X[sample])
                    y_rand.append(y[sample])
                X_rand = np.array(X_rand)
                y_rand = np.array(y_rand)
            else:
                X_rand = X
                y_rand = y

            # Fit the decision tree on the weighted data
            tree.fit(X_rand, y_rand)
            
            # Calculate weighted error
            y_pred = tree.predict(X)
            err = 0
            for j in range(len(y)):
                if (y_pred[j] != y[j]):
                    err += w[j]
            
            # Calculate classifier weight
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            # Flip the prediction on classifiers with bad performance
            if (err > 0.5): alpha = -alpha
            self.alphas.append(alpha)
            self.classifiers.append(tree)

            # Renew weight
            for j in range(len(y_pred)):
                if (y_pred[j] == 0):
                    y_pred[j] = -1
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        # Initialize predictions
        predictions = np.zeros(X.shape[0])

        # Sum the prediction of classifiers according to weight
        for alpha, clf in zip(self.alphas, self.classifiers):
            pred = clf.predict(X)
            for i in range(len(pred)):
                if pred[i] == 0:
                    pred[i] = -1
            predictions += alpha * pred
        
        # Interchange the output for negative to 0
        res = np.sign(predictions)
        for i in range(len(res)):
            if res[i] == -1:
                res[i] = 0
        return res
    


# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    # tree = DecisionTree(criterion='gini', max_depth=15)
    # tree.fit(X_train, y_train)
    # tree.plot_feature_importance_img(["age","sex","cp","fbs","thalach","thal"])
    # y_pred = tree.predict(X_train)
# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='entropy', n_estimators=33)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    

# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0
        e = self.iteration
        lr = self.learning_rate
        hypth = np.zeros(y.shape)
        testNum = y.shape[0]

        for i in range(e):
            for j in range(testNum):
                hypth[j] = self.hypoth(X[j])

            J_W = np.dot(np.transpose(X), (hypth - y)) / testNum
            J_b = np.sum((hypth - y)) / testNum

            self.weights = self.weights - lr * J_W
            self.intercept = self.intercept - lr * J_b
            #if (i%10 == 0): print(hypth)
        return
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            prediction[i] = self.hypoth(X[i])
            if prediction[i] > 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
        return prediction
    
    def hypoth(self, X):
        H = self.sigmoid(np.dot(self.weights.T, X)+self.intercept)
        return H
    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        s = 1/(1 + np.exp(-x))
        return s
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None


    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        f, t = X.shape
        X0 = X[y == 0]
        X1 = X[y == 1]
        u0 = np.sum(X0, axis=0) / X0.shape[0]
        u1 = np.sum(X1, axis=0) / X1.shape[0]
        
        Sw0 = np.dot((X0 - u0).T, (X0 - u0))
        Sw1 = np.dot((X1 - u1).T, (X1 - u1))
        Sw = Sw0 + Sw1

        mean_diff = (u1 - u0).reshape((-1, 1))
        Sb = np.dot(mean_diff, mean_diff.T)
        w = np.dot(np.linalg.inv(Sw), (u1 - u0)).flatten()

        self.m0 = u0.reshape(w.shape)
        self.m1 = u1.reshape(w.shape)
        self.sw = Sw
        self.sb = Sb
        self.w = w
        self.slope = w[1] / w[0]

        return

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i].reshape(self.w.shape)
            p = np.dot(x, self.w.T)
            D0 = np.dot(self.m0, self.w.T)
            D1 = np.dot(self.m1, self.w.T)
            if (abs(D0 - p) > abs(D1 - p)):
                y[i] = 1
        return y

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        slope = self.slope # Slope of the line
        intercept = 320  # You can choose any value for better visualization

        pts = np.linspace(40, 65)
        # Step 1: Plot the projection line trained on the training set
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=self.predict(X), cmap='viridis')
        plt.plot(pts, [slope * x + intercept for x in pts], label='Prediction Line')
        for x, y in X:
            m = -np.reciprocal(slope)
            b = y - m*x
            zx = (b - intercept) / (slope - m)
            zy = m * zx + b
            plt.annotate('', (zx, zy), (x, y), arrowprops=dict(linewidth=0.5,arrowstyle='-',color='lightblue'))
        plt.title(f'Projection Line: Slope={slope}, Intercept={intercept}')
        plt.xlabel('age')
        plt.ylabel('thalach')
        plt.legend()
        plt.axis('equal')
        plt.show()
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0004, iteration=25000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    # FLD.plot_projection(X_test)


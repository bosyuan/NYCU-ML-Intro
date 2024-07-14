# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.gradient_prediction_history = None
        self.epoch = None
        self.drop = 0.1
        self.epochs_drop = 80
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        N, M = X.shape
        
        N1 = np.ones((N, 1))
        X = np.concatenate((X, N1), axis=1)

        XTrans = np.transpose(X)
        XMult = np.matmul(XTrans, X)
        XMult = np.linalg.inv(XMult)
        XMult = np.matmul(XMult, XTrans)
        Beta = np.matmul(XMult, y)
        self.closed_form_weights = Beta[:M]
        self.closed_form_intercept = Beta[M]
        pass

    # This function computes the gradient descent solution of linear regression.      
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        N, M = X.shape
        M = M+1
        beta = np.zeros((M,))
        N1 = np.ones((N, 1))
        X = np.concatenate((X, N1), axis=1)
        self.gradient_prediction_history = []
        self.epoch = epochs

        for _ in range(epochs):
            # print(_+1, "th epoch, current weight: \n", beta)
            # step decay control on learning rate with hyper parameters "$drop", "$epoch_drop"
            l = lr * np.power(self.drop, np.floor((1+_)/self.epochs_drop))
            for i in range(X.shape[0]):
                idx = np.random.randint(0, X.shape[0])
                xi = X[i]
                yi = y[i]
                predict = np.dot(xi, beta.T)
                
                error = yi - predict  
                gradient = -2 * error * xi

                beta -= l * gradient
            self.gradient_descent_weights = beta[:M-1]
            self.gradient_descent_intercept = beta[M-1]
            p = self.gradient_descent_evaluate(X[:, :M-1], y)
            self.gradient_prediction_history.append(p)
        pass        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        # Return the value.
        N = prediction.shape[0]
        error = prediction - ground_truth
        mse = sum(np.square(error))/N
        return mse

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        # Return the prediction.
        mx = X @ self.closed_form_weights.T
        b = np.zeros((mx.shape[0],))
        for i in range(mx.shape[0]):
            b[i] = self.closed_form_intercept
        y_pred = mx + b
        return y_pred


    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        # Return the prediction.
        mx = X @ self.gradient_descent_weights.T
        b = np.zeros((mx.shape[0],))
        for i in range(mx.shape[0]):
            b[i] = self.gradient_descent_intercept
        y_pred = mx + b
        return y_pred
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        x = [i for i in range(self.epoch)]
        y = self.gradient_prediction_history
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("MSE error")
        plt.title("SGD Linear Regression")
        plt.grid()
        plt.show()
        plt.savefig('result.jpg')


# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.0001, epochs=250)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    # LR.plot_learning_curve()
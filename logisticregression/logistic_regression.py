import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments: 
    z -- A scalar or numpy array of any size.

    Return: 
    s -- sigmoid(z)
    """

    return 1 / (1 + np.exp(-z))


class LogisticRegression():

    def __init__(self, learning_rate, num_iterations, random_state=777):
        """
        Argements:
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state


    def initialize_weights(self, dim):
        """
        This function creates a vector of weights of shape (dim, 1) for w and initializes b.
    
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
    
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias) of type float
        """

        random_gen = np.random.RandomState(self.random_state)

        self.w = random_gen.rand(dim, 1) 
        self.b = random_gen.randn()

        return self


    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)
        
        A = sigmoid(np.dot(self.w.T, X) + self.b)  
        
        for i in range(A.shape[1]):
            
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
    
        return Y_prediction


    def predict_proba(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)
        
        A = sigmoid(np.dot(self.w.T, X) + self.b)  
        
        for i in range(A.shape[1]):
            Y_prediction[0, i] = A[0, i]
    
        return Y_prediction


    def propagate(self, X, Y):
        """
        Arguments:
        X -- data of size
        Y -- true "label" vector (containing 0 and 1) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        
        m = X.shape[1]
        
        # compute activation
        A = sigmoid(np.dot(self.w.T, X) + self.b) 

        # compute cost
        cost = -1 / m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A), axis = 1, keepdims = True)
        
        dw = 1 / m * np.dot(X, (A - Y).T)
        db = 1 / m * np.sum(A - Y)
        
        cost = np.squeeze(np.array(cost))

        grads = {
            "dw": dw,
            "db": db
        }
        
        return grads, cost

    def fit(self, X, Y):
        """
        Make fit for model object 
        Return: 
        Model with trained weights
        """

        self.initialize_weights(X.shape[0])

        self.costs_log = []
    
        for _ in range(self.num_iterations):

            # Retrieve derivatives from grads
            grads, cost = self.propagate(X, Y)

            self.costs_log.append(cost)
            
            self.w = self.w - self.learning_rate * grads["dw"] 
            self.b = self.b - self.learning_rate * grads["db"]
        
        return self



if __name__ == '__main__':

    X_data, y_data = make_classification(
        n_samples=10000, n_features=20, 
        n_classes=2, n_informative=20, 
        n_redundant=0,random_state=42
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_train, X_test = X_train.T, X_test.T

    model = LogisticRegression(learning_rate=0.1, num_iterations=100)

    model.fit(X_train, Y_train)

    print(accuracy_score(Y_test, model.predict(X_test)[0]))
    print(roc_auc_score(Y_test, model.predict_proba(X_test)[0]))

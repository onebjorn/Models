import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from .params import initialize_parameters, update_parameters
from .costs import compute_cost
from .forward import L_model_forward
from .backward import L_model_backward


class NeuralNetwork():

    def __init__(self, layers_dims, random_state=1):
        self.layers_dims = layers_dims
        self.seed = random_state

    def fit(self, X, y, learning_rate=0.0075, num_iterations=3000, print_cost=False):

        np.random.seed(self.seed)
        
        # Initializing parameters 
        self.parameters = initialize_parameters(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            A1, caches = L_model_forward(X, self.parameters)

            # Compute cost.
            cost = compute_cost(A1, y)
        
            # Backward propagation.
            grads = L_model_backward(A1, y, caches)
            
            # Update parameters.
            self.parameters = update_parameters(self.parameters, grads, learning_rate)
                    
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        return self

    def predict(self, X):
        """
        This function is used to predict the results of a L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label

        Returns:
        p -- list of predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = L_model_forward(X, self.parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        return p[0,:]


    def predict_proba(self, X):
        """
        This function is used to predict the results of a L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label

        Returns:
        p -- list of probability predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = L_model_forward(X, self.parameters)

        return probas


if __name__ == '__main__':

    X_data, y_data = make_classification(
        n_samples=10000, n_features=20, 
        n_classes=2, n_informative=20, 
        n_redundant=0,random_state=42
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    n_y = 1
    layers_dims = (20, 10, n_y)

    model = NeuralNetwork(layers_dims)

    model.fit(X_train.T, Y_train, learning_rate=0.015, num_iterations=3500, print_cost=True)

    print(accuracy_score(Y_test, model.predict(X_test.T)))
    print(roc_auc_score(Y_test, model.predict_proba(X_test.T)[0,:]))

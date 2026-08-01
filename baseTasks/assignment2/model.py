import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for p in self.params().values():
            p.grad = np.zeros_like(p.grad)
        
        output = self.fc1.forward(X)
        output = self.relu.forward(output)
        output = self.fc2.forward(output)
        loss, doutput = softmax_with_cross_entropy(output, y)

        doutput = self.fc2.backward(doutput)
        doutput = self.relu.backward(doutput)
        _ = self.fc1.backward(doutput)
        
        # After that, implement l2 regularization on all params
        for k, p in self.params().items():
            if k[-1] == 'w':
              l2_loss, l2_grad = l2_regularization(p.value, self.reg)
              p.grad += l2_grad
              loss += l2_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        output = self.fc1.forward(X)
        output = self.relu.forward(output)
        output = self.fc2.forward(output)
        pred = np.argmax(output, axis=1)
        return pred

    def params(self):
        result = {}
  
        result = {
          'fc1_w': self.fc1.params()['W'], 
          'fc1_b': self.fc1.params()['B'], 
          'fc2_w': self.fc2.params()['W'], 
          'fc2_b': self.fc2.params()['B'], 
          }

        return result

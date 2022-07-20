from torch import nn


class BaseNeuralNetwork(nn.Module):
    """
    A wrapper for every neural network that is in this environment
    """

    def get_parameter_count(self):
        """
        Method returns the total number of learnable parameters

        :return: count of parameters
        """
        pass

    def get_layer_count(self):
        """
        This method returns the number of layers in the network

        :return: count of network layers
        """
        pass

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        Thie method must return the state of the model

        :return: state dict
        """
        pass

    def parameters(self, recurse: bool = True):
        """
        This method returns the set of parameters

        :param recurse:
        :return: networks parameters
        """
        pass

    def to(self, device):
        """
        This method moves the NN to a specific device

        :param device: the device (cpu, cuda, mps) to work on
        """
        pass

    def forward(self, x):
        """
        Forward pass of the smart automatic network

        :param x: input tensor
        :return y: output tensor
        """
        pass

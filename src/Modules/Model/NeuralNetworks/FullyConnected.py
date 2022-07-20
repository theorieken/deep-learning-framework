import importlib
import sys
import traceback
import torch
from src.utils import Logger
from ..BaseNeuralNetwork import BaseNeuralNetwork


class AutomaticNetwork(BaseNeuralNetwork):
    """
    This network is fed by the job description files. Please do not change
    this code!
    """

    # Attribute stores list of nn modules of the network
    layers = None

    # A dict containing the original network description
    description = None

    # The device that this model is supposed to work on
    device = None

    def __init__(self, description: dict):
        """
        Constructor method of the cap assistant network

        :param description: json description of the network
        """
        # Call super class constructor
        super().__init__()

        # Initiate the device with CPU
        self.device = "cpu"

        # Initiate the network with
        self.layers = None

        # Save the description in case it is needed later
        self.description = description

        # create this network based on the network description
        self._build_layers()

    def to(self, device):
        """
        This method moves the NN to a specific device

        :param device: the device (cpu, cuda, mps) to work on
        """
        # Overwrite the current device
        self.device = device

    def forward(self, x):
        """
        Forward pass of the smart automatic network

        :param x: input tensor
        :return y: output tensor
        """

        # Initiate the network
        network = self._get_network()

        # Compute the output tensor of this network
        x = network(x)

        # Return the output tensor
        return x

    def get_parameter_count(self):
        """
        Method returns the total number of learnable parameters

        :return: count of parameters
        """
        return sum(p.numel() for p in self._get_network().parameters() if p.requires_grad)

    def get_layer_count(self):
        """
        This method returns the number of layers in the network

        :return: count of network layers
        """
        return len(self.layers)

    def parameters(self, recurse: bool = True):
        """
        This method returns the set of parameters

        :param recurse:
        :return: networks parameters
        """
        network = self._get_network()
        return network.parameters(recurse)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        Thie method must return the state of the model

        :return: state dict
        """
        return self._get_network().state_dict()

    def _get_network(self):
        """
        Method returns a network sequential based on the layers

        :return: network as sequential
        """

        # Create a sequential for the network
        network = torch.nn.Sequential()
        for part in self.layers:
            network.add_module(part['name'], part['module'])

        # Move the network to the desired device
        network.to(self.device)

        # Return the network
        return network

    def _build_layers(self):
        """
        This method takes the json network description and builds a network of it

        :return: pytorch sequential containing all layers
        """

        try:

            # Create a sequential neural network
            self.layers = []

            # Iterate through the desired layers and build them
            for i, layer in enumerate(self.description['layers']):

                # Try to obtain and construct the module
                layer_name = layer.pop('name', 'Module')
                module = importlib.import_module("torch.nn")
                instance = getattr(module, layer_name)
                layer_obj = instance(**layer)

                # Create a name for the tmp module
                sequential_module_name = layer_name.lower() + '_' + str(i + 1)

                # Add the layer object to the sequential
                self.layers.append({
                    'name': sequential_module_name,
                    'module': layer_obj
                })

        except Exception as error:

            # Log the failed generation
            Logger.log("Network generation failed: " + str(error), type="ERROR")

            # Print more information about the error
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info

            # Raise final exception to end process
            raise Exception("Network generation has failed")
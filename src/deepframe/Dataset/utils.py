"""
This file contains the data transformer, an instance that applies the
specified transformations to the data samples
"""

import importlib
import copy
from deepframe.utils import Logger


class BaseTransform(object):

    prior_knowledge = None

    def __init__(self):
        # Initialize prior knowledge
        self.prior_knowledge = {}

    def inject_knowledge(self, knowledge):
        self.prior_knowledge = knowledge


class DataTransformer(object):
    # Random center coordinates
    random_center = None

    # Dict containing injected organ centers
    organ_centers = None

    output_mode = None
    loading_mode = None

    prior_knowledge = None

    def __init__(self, transforms):
        """
        Constructor of the data transformer

        :param transforms:
        """

        self.prior_knowledge = {}

        transformer_name = None
        checked_transforms = []
        for i, transform in enumerate(transforms):
            if type(transform) is dict:

                # Produce a care package with transform specification
                care_package = copy.deepcopy(transform)

                try:

                    # Check if the name is specified (if not, continue to next transform)
                    if 'name' not in transform:
                        Logger.log("Index 'name' in dataset transforms missing (skipping transform {:d})".format(i), type="ERROR")
                        continue

                    # Cut out the name of the transformer
                    transformer_name = care_package.pop('name')

                    # Check for module in model distributions
                    module = importlib.import_module('src.Modules.Dataset.transforms')

                    # Load the model from the distributions
                    transformer = getattr(module, str(transformer_name))

                    # Append this included transformer to the list
                    checked_transforms.append(transformer(**care_package))

                except:

                    try:

                        # Try to load the transformer from torchvision
                        torchvision = importlib.import_module('torchvision.transforms')
                        tv_transformer = getattr(torchvision, str(transformer_name))

                        # If successful load, append to checked_transforms
                        checked_transforms.append(tv_transformer(**care_package))

                    except:

                        Logger.log("Transform {} not found in transforms and torchvision".format(transformer_name), type="ERROR")
                        continue

            else:

                # If this is already a transform, go
                checked_transforms.append(transform)

        # Save the transforms for later operation
        self.transforms = checked_transforms

    def __call__(self, data):

        # Iterate through all transformers and apply them
        for t in self.transforms:

            # Inject knowledge
            t.inject_knowledge(self.prior_knowledge)

            # Call transform
            data = t(data)

        # Return the finally transformed data set
        return data

    def inject_knowledge(self, key, value):
        self.prior_knowledge[key] = value

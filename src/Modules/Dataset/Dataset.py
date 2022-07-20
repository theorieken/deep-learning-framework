class Dataset:
    """
    A wrapper class for data set for this environment
    """

    # Overwrite this in the child class and assign a list with needed keys
    needed_keys = None

    def __init__(self, *args, **kwargs):
        """
        Shared constructor method of every dataset class that derives
        from this data set class.

        :param args:
        :param kwargs:
        """
        # Check the needed keys
        self._check_for_needed_keys(kwargs)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def _check_for_needed_keys(self, data):
        """
        This method checks the incoming data to **kwargs against
        a list of keys that are expected by the system

        :param data: dict based on __init__(**kwargs)
        :raise exception: if a certain key is missing
        """
        for k in self.needed_keys:
            if k not in data:
                raise Exception("The key " + str(k) + " is missing in JSON specification of the dataset")

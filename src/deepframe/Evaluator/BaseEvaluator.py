"""
This file contains the evaluator class which evaluates the trained model after training
or ad hoc.
"""


class BaseEvaluator:
    """
    This is the base evaluator class that implements some general methods that every evaluator shall have
    """

    def evaluate(self, trained_model, evaluation_set, job_path):
        """
        This method is abstract and has to be implemented by

        :param trained_model:
        :param job_path:
        :param evaluation_set:
        :return: the score of evaluation
        """
        pass

"""
This file contains metric loggers that can be added in a job configuration. They are called
after each epoch to enable the user to log custom metrics e.g. to WANDB or the filesystem.
"""


class BaseMetricLogger:
    """
    This is the base evaluator class that implements some general methods that every evaluator shall have
    """

    def single_eval_batch_operations(self, epoch, model, inputs, labels, log_callback, job_folder):
        """
        This method is called once for each mini batch in the evaluation dataset. Use this function
        to e.g. compute own averages or KPIs over the eval mini batches. You can use attributes in your
        metrics logger classes to access data.

        :param epoch: the current epoch of the logger being called
        :param model: the current model during training
        :param inputs: the last inputs in the epoch
        :param labels: the last labels in the epoch
        :param log_callback: wandb worker to log metrics there
        :param job_folder: folder where job data / stats / metrics can be logged
        """
        pass

    def log_metrics(self, epoch, model, inputs, labels, log_callback, wandb_worker, job_folder):
        """
        With this method, the user can create own metrics and log them. If you use the wb_worker,
        make sure NOT TO COMMIT the log message. The runner will log the metrics all at once later.
        This method is called AFTER the evaluation loop and can be used to log metrics or work with
        local class data. If you want to perform operations foreach mini batch, use the method
        self.single_eval_batch_operations()

        :param epoch: the current epoch of the logger being called
        :param model: the current model during training
        :param inputs: the last inputs in the epoch
        :param labels: the last labels in the epoch
        :param log_callback: callback function to log stuff
        :param wandb: wandb worker to log metrics there
        :param job_folder: folder where job data / stats / metrics can be logged
        """
        pass

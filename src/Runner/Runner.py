"""
This file contains the runner class which runs jobs (input as path to json)
"""
import pandas as pd

from ..utils import Logger, Timer, bcolors
from src.Modules.Model.NeuralNetworks.FullyConnected import AutomaticNetwork
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from torch import autograd
import torch
import ssl
import traceback
import wandb
import sys
import importlib
import json
import os


class Runner:
    """
    This trainer instance performs the training of the network by executing jobs
    """

    # Attribute stores an instance of the network
    model = None

    # A data set that will overwrite the data set specified in the job
    dataset = None

    # The path to the current jobs directory (in ./jobs/<job_name>)
    path = None

    # Attribute will contain the last checkpoint if exists (else none)
    checkpoint = None

    # An instance of a timer to measure the performances etc.
    timer = None

    # The current train data set that is used by the runner
    train_data = None

    # The current evaluation data set that is used by the runner
    eval_data = None

    # Shows whether the data log file is initial
    epoch_log_initial = None
    store_log_initial = None
    step = None
    epoch_log = None

    # This attribute stores the current job the runner is working on
    job = None

    # Attribute that stores the jobs that still need to be done
    job_queue = None

    # A reference to the wandb worker
    wandb_worker = None

    def __init__(self, jobs=None, dataset=None, model=None):
        """
        Constructor of trainer where some basic operations are done

        :param jobs: a list of json files to be executed
        :param debug: when debug mode is on, more status messages will be printed
        :param dataset: A data set that will overwrite the data set specified in the job
        """

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Fixing possible SSL errors
        ssl._create_default_https_context = ssl._create_unverified_context
        # A data set that will overwrite the data set specified in the job
        self.dataset = dataset
        self.model = model

        # Initialize eval and train data variables
        self.eval_data = None
        self.train_data = None
        self.test_data = None

        # Create a timer instance for time measurements
        self.timer = Timer()

        # Initialize the job queue
        self.job_queue = []

        # Iterate through the passed jobs
        for job in jobs:

            # Create the absolut path to the job file
            job_path = os.path.join(base_path, job)

            # Check whether the job file exist
            if os.path.exists(job_path):

                try:
                    # Open the file that contains the job description
                    f = open(job_path)

                    # Load the json data from the file
                    job_data = json.load(f)

                    # Get the job description
                    job_description = Runner._check_job_data(job_data)

                    # Append the json path for the data creator
                    job_description["json_path"] = job_path

                    # Append the job data to the job queue
                    self.job_queue.append(job_description)

                except Exception:
                    raise ValueError(bcolors.FAIL + "ERROR: Job file can not be read (" + job + ")" + bcolors.ENDC)

            else:
                # Print loading message
                print(bcolors.FAIL + "Given job path does not exist (" + job + ")" + bcolors.ENDC)

    def run(self):
        """
        This method can be called in order to run a job (encoded in json format)
        """

        # Iterate through all jobs
        for index, job in enumerate(self.job_queue):

            try:

                # Set data logging to initial
                self.epoch_log_initial = True
                self.store_log_initial = True
                self.epoch_log = pd.DataFrame()
                self.step = 1

                # Save the current job for instance access
                self.job = job

                # Obtain the base path at looking at the parent of the parents parent
                base_path = Path(__file__).parent.parent.parent.resolve()

                # A directory that stores jobs data
                job_data_dir = os.path.join(base_path, "jobs", job["name"])

                # Save this path to the runner object for now to be able to store stuff in there
                self.path = job_data_dir

                # Check if log dir exists, if not create
                Path(job_data_dir).mkdir(parents=True, exist_ok=True)

                # Create logger and clear the current file
                Logger.initialize(log_path=job_data_dir)

                # Reset the log file
                if not job["resume"]:
                    # If no resuming is desired, clear the logger
                    Logger.clear()

                # Print CLI message
                Logger.log("Started the job '" + job["name"] + "'", "HEADLINE")

                # check whether the job description has changed (if that is the case, re-run the job)
                self.job['specification_path'] = os.path.join(self.path, "specification.json")

                # Write the specification file to the job
                with open(self.job['specification_path'], "w") as fp:

                    # Save the json file out
                    json.dump(job, fp)

                # Load the last checkpoint
                self.checkpoint = self._load_checkpoint() if job["resume"] else None

                # Create an instance of the model
                self.model = self._get_model(job["model"])

                # Check if job contains index "training"
                if "training" in job and type(job["training"]) is dict:

                    # Call train method
                    self._train()

                # Check if job contains index "evaluation"
                if "evaluation" in job and type(job["evaluation"]) is dict:

                    # Call evaluation method
                    self._evaluate()

            except Exception as error:

                # print error message that this job failed
                print(bcolors.FAIL + "Fatal error occured in job: " + str(error) + bcolors.ENDC)

                # Print more information about the error
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info

            # Save the log file for this job
            if self.job["wandb_api_key"]:
                # Check if a log file exists
                if not os.path.isfile(Logger.path):
                    # Save this log file to wandb
                    self.wandb_worker.save(Logger.path)

        # Print done running message
        print(bcolors.OKGREEN + "Runner finished!" + bcolors.ENDC)

    def _init_training(self):
        """
        This method sets up weights and biases and initiates the training loop
        """

        # Check if the model has been trained in a previous run already
        if self.checkpoint is not None:

            # Check if training is already done
            if self.checkpoint["training_done"]:
                # Log that training has already been done
                Logger.log("Model is already fully trained, skipping training", type="SUCCESS")
                return True

            # Extract epoch to continue training
            start_epoch = self.checkpoint["epoch"] + 1
            wandb_id = self.checkpoint["wandb_run_id"]

        else:

            # Fallback to a default start epoch of zero
            start_epoch = 0
            wandb_id = None

        # Check if wandb shall be used
        if self.job["wandb_api_key"]:

            # Disable wandb console output
            os.environ["WANDB_SILENT"] = "true"
            wandb.login(key=self.job["wandb_api_key"])
            if start_epoch > 0 and wandb_id is not None:
                # Flash notification
                Logger.log("Loading wand and attempting to resume run " + str(wandb_id))
                self.wandb_worker = wandb.init(
                    id=wandb_id, project=self.job["wandb_project_name"], resume="allow", name=self.job["name"]
                )

            else:
                # Flash notification
                Logger.log("Loading wand for project " + self.job["wandb_project_name"])
                self.wandb_worker = wandb.init(project=self.job["wandb_project_name"], name=self.job["name"])

            # If wandb is activated, save the job configuration
            self.wandb_worker.save(self.job['specification_path'])

        return start_epoch

    def _train(self):
        """
        This method will train the network

        :param training_setup: the dict containing everything regarding the current job
        """

        # Initiate weights and biases
        start_epoch = self._init_training()

        # Start timer to measure data set
        self.timer.start("creating dataset")

        # Get dataset if not given
        dataset = Runner._get_dataset(self.job["training"]["dataset"])

        # Start timer to measure data set
        creation_took = self.timer.get_time("creating dataset")

        # Notify about data set creation
        Logger.log("Generation of data set took " + ("{:.2f}".format(creation_took)) + " seconds")

        # Get dataloader for both training and validation
        self.train_data, self.eval_data = Runner._get_dataloader(
            dataset=dataset,
            split_ratio=self.job["training"]["split_ratio"],
            num_workers=self.job["training"]["dataset"]["num_workers"],
            batch_size=self.job["training"]["batch_size"],
        )

        # Log the available device
        Logger.log("Using device: " + self._get_device() + (" (forced)" if self.job['force_cpu'] else ""), type="INFO")

        # Log dataset information
        Logger.log("Start training on " + str(len(self.train_data)) + " batches ", type="INFO")

        # Notify the user regarding validation
        Logger.log("Validation is done on " + str(len(self.eval_data)) + " batches ...")

        # Get an optimizer, scheduler and loss function
        optimizer = self._get_optimizer(self.job["training"]["optimizer"])
        scheduler = self._get_lr_scheduler(optimizer, self.job["training"]["lr_scheduler"])
        metric_logger = Runner._get_metric_logger(self.job["training"]["metric_logger"])
        loss_function = Runner._get_loss_function(self.job["training"]["loss"])

        # Enable weights and biases logging
        if self.job["wandb_api_key"]:

            # Set the watcher on the model
            self.wandb_worker.watch(self.model, log="gradients", log_freq=1)

        # Log single data points about structure
        self._log_data({
            'network_layers': self.model.get_layer_count(),
            'parameter_count': self.model.get_parameter_count(),
            'batch_size': self.job["training"]["batch_size"],
            'split_ratio': self.job["training"]["split_ratio"],
            'train_batch_count': len(self.train_data),
            'eval_batch_count': len(self.eval_data)
        }, per_epoch=False)

        # Check if start epoch is not zero and notify
        if start_epoch > 0:
            # Print notification
            Logger.log("Resuming training in epoch " + str(start_epoch + 1))

        # Warn user in case detect bad gradients has been chosen
        if self.job["training"]["detect_bad_gradients"]:
            # Flash warning about bad gradients
            Logger.log("Selected detect_bad_gradients - using AutoGrad", type="WARNING")

        # Initiate early stopping
        best_eval_loss = torch.inf
        current_eval_loss = best_eval_loss
        early_stopping_counter = 0

        # Initiate inputs and labels variable
        inputs = None
        labels = None

        # Iterate through epochs (based on jobs setting)
        for epoch in range(start_epoch, self.job["training"]["epochs"]):

            # Start epoch timer and log the start of this epoch
            Logger.log("Starting to run Epoch {}/{}".format(epoch + 1, self.job["training"]["epochs"]), in_cli=True, new_line=True)

            # Print epoch status bar
            Logger.print_status_bar(done=0, title="training loss: -")

            # Start the epoch timer
            self.timer.start("epoch")

            # Set model to train mode
            self.model.train()

            # Initialize variables
            running_loss = 0

            # Set model to device
            self.model.to(self._get_device())

            # Initiate a print out timer (to have not too small period)
            self.timer.start('train_printer')

            # Run through batches and perform model training
            for batch, batch_input in enumerate(self.train_data):

                # Separate the batch input in inputs and labels
                inputs, labels = batch_input

                # Move the data to desired device
                inputs, labels = inputs.to(self._get_device()), labels.to(self._get_device())

                # Reset gradients
                optimizer.zero_grad()

                # Compute forward path of the model
                model_output = self.model(inputs)

                # Calculate loss
                loss = loss_function(model_output, labels)

                # Check if user wants to do backpropagation with detect anomaly on (slow!)
                if self.job["training"]["detect_bad_gradients"]:
                    with autograd.detect_anomaly():
                        loss.backward()
                else:
                    # Backpropagation
                    loss.backward()

                # Perform optimization step
                optimizer.step()

                # Add loss
                running_loss += loss.detach().cpu().numpy()

                # Get the current running los
                current_loss = running_loss / (batch + 1)
                training_done_percentage = ((batch + 1) / len(self.train_data)) * 100

                # Check if minimum interval has passed
                if self.timer.get_time('train_printer', reset=False) >= 0.1 or training_done_percentage > 99.9:

                    # Print epoch status bar
                    Logger.print_status_bar(
                        done=training_done_percentage,
                        title="training loss: " + "{:.5f}".format(current_loss),
                    )

                    # Restart the timer
                    self.timer.start('train_printer')

            # Finish the status bar
            Logger.end_status_bar()

            # Calculate epoch los
            epoch_train_loss = running_loss / len(self.train_data)

            # Perform validation
            if self.eval_data is not None:

                with torch.no_grad():

                    # Print epoch status bar
                    Logger.print_status_bar(done=0, title="evaluation loss: -")

                    # Set model to evaluation mode
                    self.model.eval()

                    # Initialize a running loss of 99999
                    eval_running_loss = 0

                    # Reset timer
                    self.timer.start('eval_printer')

                    # Perform validation on healthy images
                    for batch, batch_input in enumerate(self.eval_data):

                        # Extract inputs and labels from the batch input
                        inputs, labels = batch_input

                        # Move the data to desired device
                        inputs, labels = inputs.to(self._get_device()), labels.to(self._get_device())

                        # Calculate output
                        model_output = self.model(inputs)

                        # Determine loss
                        eval_loss = loss_function(model_output, labels)

                        # Add to running validation loss
                        eval_running_loss += eval_loss.detach().cpu().numpy()

                        # Get the current running los
                        current_eval_loss = eval_running_loss / (batch + 1)
                        eval_done_percentage = ((batch + 1) / len(self.eval_data)) * 100

                        # Log metrics for every batch in the eval dataset
                        if metric_logger is not None:
                            metric_logger.single_eval_batch_operations(epoch, self.model, inputs, labels, self._log_data, self.path)

                        # Check if minimum interval has passed
                        if self.timer.get_time('eval_printer', reset=False) >= 0.1 or eval_done_percentage > 99.9:
                            # Print epoch status bar
                            Logger.print_status_bar(
                                done=eval_done_percentage,
                                title="evaluation loss: " + "{:.5f}".format(current_eval_loss),
                            )

                            # Restart the timer
                            self.timer.start('eval_printer')

                # End status bar
                Logger.end_status_bar()

                # Do early stopping here
                if current_eval_loss < best_eval_loss - 1e-2:  # Should improve at least this over n epochs
                    # Loss decreased - RESET
                    best_eval_loss = current_eval_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.job["training"].get("early_stopping_patience", torch.inf):
                        val = self.job["training"].get("early_stopping_patience", torch.inf)
                        Logger.log(f"Initialising early stopping after model not improved for {val} epochs", type="INFO")
                        break

                # Calculate epoch train val loss
                epoch_evaluation_loss = eval_running_loss / len(self.eval_data)

            else:
                # If no validation is done, we take the train loss as val loss
                epoch_evaluation_loss = epoch_train_loss

            # Also perform a step for the learning rate scheduler
            scheduler.step()

            # Obtain the current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Stop timer to measure epoch length
            epoch_time = self.timer.get_time("epoch")

            # Log the epoch success
            avg_loss = "{:.4f}".format(epoch_train_loss)
            avg_val_loss = "{:.4f}".format(epoch_evaluation_loss)

            # Log metrics via metric logger when it exists
            if metric_logger is not None:
                metric_logger.log_metrics(epoch, self.model, inputs, labels, self._log_data, self.wandb_worker, self.path)

            # Report current loss, learning rate, epoch etc.
            self._log_data({
                "Training Loss": epoch_train_loss,
                "Evaluation Loss": epoch_evaluation_loss,
                "Learning Rate": current_lr,
                "Epoch (Duration)": epoch_time,
                "Epoch": epoch + 1,
            })

            # Save a checkpoint for this job after each epoch (to be able to resume)
            self._save_checkpoint({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "train-loss": epoch_train_loss,
                "eval_loss": epoch_evaluation_loss,
                "training_done": epoch == (self.job["training"]["epochs"] - 1),
                "wandb_run_id": self.job["wandb_run_id"],
            })

            # Obtain the current learning rate
            lr_formatted = "{:.6f}".format(current_lr)

            # Print epoch status
            Logger.log("Train loss: " + avg_loss + ". Eval loss: " + avg_val_loss + ". LR: " + lr_formatted)
            Logger.log("Epoch took " + str(epoch_time) + " seconds.")

        # Write log message that the training has been completed
        Logger.log("Training of the model completed", type="SUCCESS")

        # Check if wandb shall be used
        if self.job["wandb_api_key"] and self.wandb_worker is not None:
            self.wandb_worker.finish()

    def _evaluate(self):
        """
        This method will evaluate the network

        :param evaluation_setup: the dict containing everything regarding the current job
        """

        # Log dataset information
        Logger.log("Start evaluation of the model", type="INFO")

        # Extract the evaluation setup
        evaluation_setup = self.job["evaluation"]

        # Get the test data set
        dataset = Runner._get_dataset(self.job["evaluation"]["dataset"])

        # Get the test data loader
        self.test_data, _ = Runner._get_dataloader(dataset=dataset, batch_size=self.job["evaluation"]["batch_size"])

        # Get the evaluation instance
        evaluator = Runner._get_evaluator(evaluation_setup)

        # Call evaluate on the evaluator
        evaluator.evaluate(self.model, self.test_data, self.path)

        # Write log message that the training has been completed
        Logger.log("Evaluation of the model completed", type="SUCCESS")

    def _load_checkpoint(self):
        """
        Returns the checkpoint dict found in path.

        TODO: map location could be switched to possible GPUs theoretically
        """

        # Generate a variable that stores the checkpoint path
        checkpoint_path = os.path.join(self.path, "checkpoint.tar")

        # Check if the file exists
        if not os.path.exists(checkpoint_path):
            return None

        # Load the checkpoint
        return torch.load(checkpoint_path, map_location=torch.device("cpu"))

    def _save_checkpoint(self, checkpoint_dict):
        """
        Saves a checkpoint dictionary in a tar object to load in case this job is repeated
        """
        save_path = os.path.join("results", self.path, "checkpoint.tar")
        torch.save(checkpoint_dict, save_path)

        # Check if wandb shall be used
        if self.job["wandb_api_key"]:
            self.wandb_worker.save(save_path)

    def _get_model(self, model_setup: dict):
        """
        This method returns the model required for the job.

        :param model_setup: the model to be selected
        :return: a model instance
        """

        # Extract the model name from the setup
        model_choice = model_setup.pop("name", None)

        # Try to import local custom module
        try:

            # Check if there is a model choice, otherwise fallback to automatic network
            if model_choice is None:
                Logger.log("Using the automatic network as fallback")

            # Check for module in model distributions
            module = importlib.import_module('src.Modules.Model.Distributions')

            # Load the model from the distributions
            model = getattr(module, str(model_choice))

            # Create an instance of the desired model
            model_instance = model(**model_setup)

        # In case the model does not exist
        except:

            if model_choice is None:
                Logger.log("Model {} could not be loaded. Using AutomaticNetwork as fallback.".format(model_choice), type="ERROR")

            # Use the smart model builder as fallback method
            model_instance = AutomaticNetwork(model_setup)

        # Check if checkpoint exists
        if self.checkpoint is not None:
            # Recover model from last
            model_instance.load_state_dict(self.checkpoint["model"])

            # Log a status message about recovery of model
            Logger.log("Recovered model from the last checkpoint", type="WARNING")

        return model_instance

    def _get_optimizer(self, optimizer_setup: dict, **params):
        if optimizer_setup["name"] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer_setup["learning_rate"])
            if self.checkpoint is not None:
                Logger.log("Recovering optimizer Adam from the last checkpoint", type="WARNING")
                try:
                    optimizer.load_state_dict(self.checkpoint["optimizer"])
                except Exception:
                    Logger.log("Could not recover optimizer checkpoint state", type="ERROR")
            return optimizer
        else:
            raise ValueError(
                bcolors.FAIL
                + "ERROR: Optimizer "
                + optimizer_setup["name"]
                + " not recognized, aborting"
                + bcolors.ENDC
            )

    def _get_lr_scheduler(self, optimizer, scheduler_setup: dict):
        """
        Returns a scheduler for the learning rate

        :param optimizer:
        :param scheduler_setup:
        :return:

        TODO: make this dynamic
        """

        # Every scheduler will need the optimizer
        scheduler_setup["optimizer"] = optimizer
        scheduler_name = scheduler_setup.pop("name")
        module = importlib.import_module("torch.optim.lr_scheduler")
        lr_scheduler_class = getattr(module, scheduler_name)
        scheduler = lr_scheduler_class(**scheduler_setup)

        # Check if there is a checkpoint
        if self.checkpoint is not None:
            Logger.log("Recovering scheduler " + scheduler_name + " from the last checkpoint", type="WARNING")

            # Load the state dict
            try:
                scheduler.load_state_dict(self.checkpoint["lr_scheduler"])
            except Exception:
                Logger.log("Could not recover scheduler checkpoint state", type="ERROR")

        # Return the scheduler
        return scheduler

    def _get_device(self):

        # If force CPU, use CPU anyways
        if self.job['force_cpu']:
            return "cpu"

        # Try to use the new torch version
        try:
            if torch.cuda.is_available():
                return "cuda:" + str(self.job['gpu'])
            elif torch.backends.mps.is_available():
                return "mps:" + str(self.job['gpu'])
            else:
                return "cpu"

        # Otherwise use the standard
        except:
            if torch.cuda.is_available():
                return "cuda:" + str(self.job['gpu'])
            else:
                return "cpu"

    def _log_data(self, data: dict, commit: bool = True, per_epoch: bool = True):
        """
        This method stores data in the job directory

        :param data: a dict storing data
        :param commit: whether to write the data now
        :param per_epoch:
        """

        # Create paths for both files
        data_log_file = os.path.join(self.path, 'epoch_data.csv' if per_epoch else 'store_data.txt')

        # Inline function that transforms dict to keys and values
        def key_value_from_dict(data, name=""):
            keys = []
            values = []
            if type(data) is dict:
                for key in data.keys():
                    if key == "step":
                        raise ValueError("You can not use 'step' as a key when logging data")
                    child_header, child_data = key_value_from_dict(data[key], name + '.' + key if name != "" else key)
                    keys += child_header
                    values += child_data
            else:

                # FIXME: here, only float, int and str can be appended. Other things (e.g. wandb histograms shall not!)
                keys.append(name)
                values.append(data)
            # Add the step to this data
            if name == "":
                keys.append('step')
                values.append(self.step)
            return keys, values

        # Transform the incoming data to csv-able data
        keys, values = key_value_from_dict(data)

        # Check if this is epoch or general data
        if per_epoch:

            # compose data from from values
            frame = pd.DataFrame(columns=keys, data=[values])

            # If we are initial and there is already a frame, it is old!
            if self.epoch_log_initial:
                if os.path.exists(data_log_file):
                    # FIXME: maybe a logging warning?
                    os.remove(data_log_file)

            # merge the existing dataframe and the new one
            if not self.epoch_log.empty:
                frame = pd.merge(self.epoch_log, frame, on='step')

            # Check if this data shall be written now or wait
            if commit:

                # Obtain the old frame with existing log data
                old_frame = pd.DataFrame()
                if os.path.exists(data_log_file):
                    # FIXME: there should be a better way
                    try:
                        old_frame = pd.read_csv(data_log_file)
                    except:
                        Logger.log("Local CSV data can't be updated", "WARNING")
                if not old_frame.empty:
                    self.epoch_log = pd.concat((old_frame, frame))

                # write the current data frame now
                self.epoch_log.to_csv(data_log_file, index=False)
                self.epoch_log = pd.DataFrame()
                self.step += 1
            else:
                # store the values in the current data frame
                self.epoch_log = frame

            # Set the initial flag to false
            self.epoch_log_initial = False

        else:

            # just write the data in a file
            with open(data_log_file, "w" if (self.store_log_initial and not self.job["resume"]) else "a") as file:
                for i, key in enumerate(keys):
                    # Do not write the step when no epoch data to merge
                    if key != "step":
                        file.write(key + ': ' + str(values[i]) + "\n")

        # Also log to wandb if setup
        if self.wandb_worker:
            self.wandb_worker.log(data, commit=commit)

    @staticmethod
    def _get_loss_function(loss_function_setup):
        try:
            module = importlib.import_module("src.losses")
            loss_class = getattr(module, loss_function_setup["name"])
            loss_fun = loss_class(**loss_function_setup)
        except:
            module = importlib.import_module("torch.nn")
            loss_class = getattr(module, loss_function_setup["name"])
            loss_fun = loss_class()

        return loss_fun

    @staticmethod
    def _get_metric_logger(metric_logger_setup):
        """
        Method constructs a metric logger (based on json description) or returns
        None if there is nothing specified or faulty

        :param metric_logger_setup: dict with setup
        :return: metric logger instance as callable
        """

        try:

            # Try to load a metric logger from logger distributions
            module = importlib.import_module("src.Modules.MetricLogger.Distributions")
            logger_class = getattr(module, metric_logger_setup["name"])

            # Log usage of this logger
            Logger.log("Runner will use metrics logger " + metric_logger_setup["name"])

            # Return logger instance to runner
            return logger_class(**metric_logger_setup)

        except Exception as error:

            # Check if there was a logger specified and flash error then
            if 'name' in metric_logger_setup:
                Logger.log("Metric logger could not be built: " + str(error), type="ERROR")

        # Fallback return is none
        return None

    @staticmethod
    def _check_job_data(job_data: dict):
        """
        This method checks whether a passed job (in terms of a path to a json file) contains everything needed

        :param job_data: a dict that stores all job data
        :return: job data is okay and contains everything
        """

        # TODO: implement this tests and default autocomplete later (prioritizing!)

        # TODO: flash warnings when specific parts of the job description are missing and defaults are used

        # Add a default scheduler
        job_data["training"].setdefault(
            "lr_scheduler",
            {"name": "LinearLR", "start_factor": 1, "end_factor": 0.01, "total_iters": 100},
        )

        # Set labels to none by default so the data set figures out the order
        job_data["training"]["dataset"].setdefault("labels", None)

        # If there is no wandb key, just ignore wandb
        job_data.setdefault("wandb_api_key", None)

        # Append a wandb id if none exists yet
        job_data.setdefault("wandb_run_id", wandb.util.generate_id())

        # Set default GPU device
        job_data.setdefault("gpu", 0)
        job_data.setdefault("force_cpu", False)

        return job_data

    @staticmethod
    def _get_evaluator(evaluation_setup: dict):
        """
        This method will return an evaluator that is specified in the jobs json file

        :param evaluation_setup:
        :return: evaluator instance
        """

        # Try to load the specified evaluator from evaluator distributions
        try:

            module = importlib.import_module("src.Modules.Evaluator.Distributions")
            evaluater_class = getattr(module, evaluation_setup["name"])
            return evaluater_class()

        # Check if evaluator does not exist in distributions
        except Exception as error:
            raise error

    @staticmethod
    def _get_dataset(description):
        """
        Method creates the data set instance and returns it based on the data (contains job description)

        :return: Dataset instance that contains samples
        """

        # Try to load the specified data set from distributions
        try:
            # TODO: keep preload or drop it?
            module = importlib.import_module("src.Modules.Dataset.Distributions")
            dataset = getattr(module, description['name'])
            return dataset(**description)

        # Check if dataset does not exist in distributions
        except Exception as error:
            raise error

    @staticmethod
    def _get_dataloader(dataset, shuffle: bool = True, split_ratio: float = None, num_workers: int = 0, batch_size: int = 64, pin_memory: bool = False):
        """
        The method returns data loader instances (if split) or just one dataloader based on the passed dataset

        :param dataset: the data set that the data loader should work on
        :param shuffle: whether the data shall be shuffled
        :param split_ratio: the ratio that the split shall be based on (if none, no split)
        :param num_workers: number of workers for laoding data
        :param batch_size: batch size of returned samples
        :param pin_memory: speeds up data loading on GPU
        :return:
        """

        # Initialize the second split (as it might be none)
        second_split = None

        # Check whether the user wants a split data set
        if split_ratio is not None:

            # Determine split threshold and perform random split of the passed data set
            split_value = int(split_ratio * len(dataset))
            first_split, second_split = random_split(dataset, [split_value, len(dataset) - split_value], generator=torch.Generator().manual_seed(10))

            # Initialize data loaders for both parts of the split data set
            first_split = DataLoader(first_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            second_split = DataLoader(second_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        else:

            # Just return one data loader then
            first_split = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        # Return tuple of splits
        return first_split, second_split

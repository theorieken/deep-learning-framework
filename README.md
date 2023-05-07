# Deep Learning Framework 

## Getting Started

This is a simple and concise framework for training neural networks and neatly storing checkpoints, logs and syncing
training progress with *Weights and Biases*. You specify a `run` with a dedicated `json file`. The path to a json file
(or a list of json files) is then passed to a `Runner` which executes the specified jobs. 

## Launch a Project

1) Run `pip install deep-learning-framework==0.1.4` to install this framework
2) Create a job configuration file (or get started with the sample in `~/examples`)
3) Create a `main.py` file as depicted below and run the job file  

```python
from src.deepframe.Runner import Runner

# Add all the jobs, that you want to run, here
jobs = ['config/<your-config>.json']

# Main guard for multithreading the runner "below"
if __name__ == '__main__':
    # Create a runner instance and pass it the jobs
    worker = Runner(jobs=jobs)

    # Start working on the jobs until all are finished
    worker.run()
```

Obviously, most often you will want to create your own models, loss functions, datasets or
metric loggers. For that reason, this framework expects your framework to have the following 
folder structure. In your `job description` file, you can specify **standard** datasets, loss 
functions, models and so on. In this case, the Runner will try to load what you specified from 
e.g. torchvision, MONAI or torch itself. If you specify the `name` attribute of e.g. a dataset
or model with the name of a file you created in the respective locations, the Runner will use
your files making this framework a home to your ideas. 

```
/your-project
│   main.py
│   ...
│
└───/config
│   │   job_01.json
│   │   job_02.json
│   │   job_03.json
│   │   runner.py
│
└───/src
│   │   losses.py
│   │   
│   └───/Models
│   │   │   CancerUNet.py
│   │   │   CompanySalesPredictor.py
│   │   │   ...
│   │
│   └───/MetricLoggers
│   │   │   CancerPredictionLogger.py
│   │   │   CompanSalesLogger.py
│   │   │   ...
│   │
│   └───/Evaluators
│   │   │   CancerUnetEvaluator.py
│   │   │   CompanySalesEvaluator.py
│   │   │   ...
│   │
│   └───/Datasets
│       │   PatientDataset.py
│       │   FullYearSalesDataset.py
│       │   ...
│
```

## Using Configuration Files

The configuration file specifies what the Runner should do. You can also add your own 
components e.g. `Loss Functions`, `Trasforms` or other `Models` in the dedicated locations.
Transforms can be also choosen from `torchvision.transform`. The runner first looks for a
matching transform in `src/Data/transforms.py`, if there is nothing with the specified name, 
it will try to import it from torchvision. If you don't choose dataset labels, the dataset
will specify an own order of labels (that will be the channels of the tensors). 

```javascript
config = {
    "name": "<name>",
    // Whether the runner should recover a model from a checkpoint
    "resume": true, 
    // Whether data shall be preloaded to speed up training (can be RAM-intensive) 
    "preload": true,
    // Weights and Biases setup
    "wandb_project_name": "<wandb project>",
    // Sample prediction slices to see current behavior logged 
    "wandb_prediction_examples": 8,
    "wandb_api_key": "<your wandb api key>",
    // Model specifications
    "model": {
        "name": "OrganNet25D",
        // Add parameters to the model here
    },
    "training": {
        "epochs": 300,
        "detect_bad_gradients": false,
        "grad_norm_clip": 1,
        // Split ratio between training and evaluation dataset
        "split_ratio": 0.77,
        "batch_size": 2,
        // Specification of the loss function 
        "loss": {
            "name": "CombinedLoss",
            // Add parameters to the loss function here
        },
        // Specification of the optimizer
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            // Add more parameters here
        },
        // Specification of the learning rate scheduler
        "lr_scheduler": {
            "name": "MultiStepLR",
            "gamma": 0.1,
            "milestones": [50, 100]
        },
        // Specification of the data set
        "dataset": {
            "root": "./data/train",
            "num_workers": 2,
            // Define the label structure globally (for reproducability)
            "labels": [
                "BrainStem",
                "Chiasm",
                "Mandible",
                "OpticNerve_L",
                "OpticNerve_R",
                "Parotid_L",
                "Parotid_R",
                "Submandibular_L",
                "Submandibular_R"
            ],
            // Transformations to be applied to the labels only
            "label_transforms": [
                {
                    "name": "Transpose",
                    "dim_1": 0,
                    "dim_2": -1
                },
                {
                    "name": "CropAroundBrainStem",
                    "width": 256,
                    "height": 256,
                    "depth": 48
                }
            ],
            // Transformations to be applied to the feature only
            "sample_transforms": [
                {
                    "name": "Transpose",
                    "dim_1": 0,
                    "dim_2": -1
                },
                {
                    "name": "CropAroundBrainStem",
                    "width": 256,
                    "height": 256,
                    "depth": 48
                },
                {
                    "name": "StandardScaleTensor"
                }
            ]
        }
    }
}
```


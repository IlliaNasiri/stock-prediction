## Motivation ##

## Description ## 

## Sliding Window ##

## Architecture ##
![Untitled Diagram drawio (2)](https://github.com/IlliaNasiri/stock-prediction/assets/135656013/b4877fc9-93d5-4fbf-b0f7-9de7e13424ee)

### config.py: ###
this file allows you to configure the parameters of the training and testing phases for the RNN stock-prediction model, as well as the visualization.

``` python

ModelConfig = { # TO BE IMPLEMENTED
    "n_features": 1,
    "n_hidden:": 64,
    "num_rnn_layers": 1,
    "dense_layers": [(64, 64), (64, 1)]
}


TrainingConfig = {
    "data_path": "data/IBM.csv", # path to the CSV file to use for training
    "attribute": "Close", # the attribute that the RNN is going to be trained on
    "window_size": 5, # Size of the sliding window
    "train_size": 0.9, # what portion of the dataset should be used for training
    "epochs": 500, # number of epochs for training
    "model_save_path": "RNN/trained_models/rnn.pth", # path where to save the model. NOTE: the folder MUST EXIST!
}

TestingConfig = {
    "data_path": "data/GOOG.csv", # path to the CSV file to use for testing
    "attribute": "close", # the attribute that the RNN is going to be tested on
    "window_size": 5, # Size of the sliding window
    "test_size": 0.0, the portion of data used for testing
    "model_save_path": "RNN/trained_models/rnn.pth", # path where to find the model
    "visualization_save_path": "RNN/visualizations/comparison.jpg" # NOTE: the folder MUST EXIST!
}

```

### model.py ###
This file contains defines the architecture of the model to be trained, as well as the training related variables that are going to be used by other scripts such as loss and the optimizer.
This is done so that the user can easily change the loss and optimizer without having to edit the train.py and test.py 
1. **RNN class:** blueprint of previously described RNN architecture 
2. **rnn_model:** an instance of RNN class that is used by train.py, test.py, and visualize.py
3. **loss:** the loss function that is used for training and testing, and is accessed by train.py and test.py
4. **optimizer:** the optimizer that is used by train.py 

### train.py ###
Once you have configured everything in the **config.py**, and have your csv dataset, you can simply run the training script. It will keep printing the loss 
every 10 iterations. Once it has been finished, it will save the model in a **.pth** file.
 
### test.py ###
Based on the **config.py**, it will print out the loss on the testing dataset.

### visualize.py ###
Based on the **config.py**, it will visualize the actual price as blue, and the predicted price as red. 

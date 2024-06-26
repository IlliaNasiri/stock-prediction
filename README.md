## Motivation ##

## Description ## 

## Sliding Window ##
To make a sequence of  data appropriate for training a model, one approach is to slide a "window" over the data. Each slide of the window will generate a new row which will be a new training example: <br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/IlliaNasiri/stock-prediction/assets/135656013/2d7f9236-a95d-4c21-8ece-1557dd0d7c9a">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/IlliaNasiri/stock-prediction/assets/135656013/da5be2a4-a16f-4788-971b-0f517db62e6b">
  <img alt="Sliding Windows Explained">
</picture>


## Architecture ##
The architecture consists of two main parts: 1. RNN layers, 2. Dense layers. <br><br>
**RNN layers:** one intuitive way to think about RNNs in my opinion is to think of it as a machine that converts (encodes) a sequence of an arbitrary length and dimensionality into a **fixed size** representation which is the hidden state. The last hidden state represents the information of the whole sequence given as a set of (n_hidden) numbers. An example would be if we fed an RNN a textbook or an email, regardless, it would still convert it to a vector of same size. Each element of this vector can potentially represent some feature of input, for example first element in the vector can correspond to whether 3 subsequent data points are increasing or decreasing, and so on...<br>

**Dense layers:** now that we have a fixed sized hidden state, we can pass it through a series of fully connected neural network layers, last of which will output a single number that represents the prediction. 

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/IlliaNasiri/stock-prediction/assets/135656013/ce6778f8-5352-4bd5-ab7a-83b8f804a755">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/IlliaNasiri/stock-prediction/assets/135656013/7df3a7e6-3e85-494b-8af1-767e50c91290">
  <img alt="Sliding Windows Explained">
</picture>


### config.py: ###
this file allows you to configure the parameters of the training and testing phases for the RNN stock-prediction model, as well as the visualization.

``` python

N_HIDDEN = 64 # Tells you to how many numbers the input is mapped to
LABEL_WIDTH = 5 # Tells how many values to predict ahead (5 next prices in this case)
WINDOW_SIZE = 15 # Tells you what is the size of the window sliding through data (window size is 15, 10 values are given for trainin, 5 are considered labels because LABEL_WIDTH = 5)

ModelConfig = {
    "n_features": 1, # Tells you the dimesionality of each element in the sequence. (for prices it's 1)
    "n_hidden:": N_HIDDEN, 
    "num_rnn_layers": 1, # tells how many stacked rnn layers 
    "dense_layers": [(N_HIDDEN, 256), (256, 512), (512, LABEL_WIDTH)] # the architecture of dense layers.
     # Note: the numbers between tuples MUST agree, e.g: (128, **256**), (**256**, 512)  
}

TrainingConfig = {
    "data_path": "data/IBM.csv", # path to the CSV file to use for training
    "attribute": "Close", # the attribute that the RNN is going to be trained on
    "window_size": WINDOW_SIZE, # Size of the sliding window
    "label_width": LABEL_WIDTH,
    "train_size": 1.0, # what portion of the dataset should be used for training
    "epochs": 300, # number of epochs for training
    "model_save_path": "RNN/trained_models/rnn.pth" # path where to save the model. NOTE: the folder MUST EXIST!
}

TestingConfig = {
    "data_path": "data/GOOG.csv", # path to the CSV file to use for testing
    "attribute": "close", # the attribute that the RNN is going to be tested on
    "window_size": WINDOW_SIZE, # Size of the sliding window
    "label_width": LABEL_WIDTH,
    "test_size": 0.0, #the portion of data used for testing
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

Example Visualization trained on IBM.csv and tested on GOOG.csv:
![comparison](https://github.com/IlliaNasiri/stock-prediction/assets/135656013/0abca124-46f6-4810-854e-860f65bf152b)



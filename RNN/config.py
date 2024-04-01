TrainingConfig = {
    "data_path": "data/IBM.csv",
    "attribute": "Close",
    "window_size": 5,
    "train_size": 0.9,
    "epochs": 500,
    "model_save_path": "RNN/trained_models/rnn.pth",
    "visualization_save_path": "RNN/visualizations/comparison.jpg"
}

TestingConfig = {
    "data_path": "data/GOOG.csv",
    "attribute": "close",
    "window_size": 5,
    "train_size": 0.0,
    "model_save_path": "RNN/trained_models/rnn.pth",
    "visualization_save_path": "RNN/visualizations/comparison.jpg"
}

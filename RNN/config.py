ModelConfig = {
    "n_features": 1,
    "n_hidden": 64,
    "num_rnn_layers": 1,
    "dense_layers": [(64, 64), (64, 1)]
}

TrainingConfig = {
    "data_path": "data/IBM.csv",
    "attribute": "Close",
    "window_size": 10,
    "train_size": 0.8,
    "epochs": 500,
    "model_save_path": "RNN/trained_models/rnn.pth"
}

TestingConfig = {
    "data_path": "data/IBM.csv",
    "attribute": "Close",
    "window_size": 10,
    "test_size": 0.2,
    "model_save_path": "RNN/trained_models/rnn.pth",
    "visualization_save_path": "RNN/visualizations/comparison.jpg"
}

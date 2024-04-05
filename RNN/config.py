ModelConfig = {
    "n_features": 1,
    "n_hidden": 64,
    "num_rnn_layers": 2,
    "dense_layers": [(64, 256), (256, 512), (512, 5)]
}

TrainingConfig = {
    "data_path": "data/IBM.csv",
    "attribute": "Close",
    "window_size": 15,
    "label_width": 5,
    "train_size": 1.0,
    "epochs": 300,
    "model_save_path": "RNN/trained_models/rnn.pth"
}

TestingConfig = {
    "data_path": "data/MSFT.csv",
    "attribute": "Close",
    "window_size": 15,
    "label_width": 5,
    "test_size": 0.1,
    "model_save_path": "RNN/trained_models/rnn.pth",
    "visualization_save_path": "RNN/visualizations/comparison.jpg"
}

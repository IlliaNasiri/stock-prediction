N_HIDDEN = 64
LABEL_WIDTH = 5
WINDOW_SIZE = 15

ModelConfig = {
    "n_features": 1,
    "n_hidden": N_HIDDEN,
    "num_rnn_layers": 2,
    "dense_layers": [(N_HIDDEN, 256), (256, 512), (512, LABEL_WIDTH)]
}

TrainingConfig = {
    "data_path": "data/IBM.csv",
    "attribute": "Close",
    "window_size": WINDOW_SIZE,
    "label_width": LABEL_WIDTH,
    "train_size": 1.0,
    "epochs": 300,
    "model_save_path": "RNN/trained_models/rnn.pth"
}

TestingConfig = {
    "data_path": "data/MSFT.csv",
    "attribute": "Close",
    "window_size": WINDOW_SIZE,
    "label_width": LABEL_WIDTH,
    "test_size": 0.1,
    "model_save_path": "RNN/trained_models/rnn.pth",
    "visualization_save_path": "RNN/visualizations/comparison.jpg"
}

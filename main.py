import tensorflow as tf
from textgenrnn import textgenrnn

train_function=textgen.train_from_file if train_cfg["line_delimited"] else textgen.train_from_largetext_file

model_cfq={
    "rnn_size": 128,
    "rnn-layers": 4,
    "rnn_bidirectional": True,
    "max_lenght": 70,
    "max_words": 1000,
    "dim_embeddings": 100,
    "word_level": False,
}

train_cfg{
    "line_delimited": False,
    "num_epochs": 10,
    "gen_epochs": 5,
    "batch_size": 128,
    "train_size": 0.87,
    "dropout": 0.0,
    "validation": False,
    "is_csv": False
}

textgen = textgenrnn()
textgen.train_from_file("Leserbriefe just txt.txt", num_epochs=4)
textgen.generate()



import tensorflow as tf
import json
import numpy as np
from data_utils import Data
from models.char_cnn_zhang import CharCNNZhang

# from models.char_cnn_kim import CharCNNKim


"""
from keras.models import load_model
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
"""
tf.flags.DEFINE_string("model", "char_cnn_zhang",
                       "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()


    # # Load model configurations and build model
    # if FLAGS.model == "kim":
    #     model = CharCNNKim(input_size=config["data"]["input_size"],
    #                        alphabet_size=config["data"]["alphabet_size"],
    #                        embedding_size=config["model"]["embedding_size"],
    #                        conv_layers=config["model"]["conv_layers"],
    #                        fully_connected_layers=config["model"]["fully_connected_layers"],
    #                        num_of_classes=config["data"]["num_of_classes"],
    #                        dropout_p=config["model"]["dropout_p"],
    #                        optimizer=config["model"]["optimizer"],
    #                        loss=config["model"]["loss"])
    # else:
    model = CharCNNZhang(input_size=config["data"]["input_size"],
                         alphabet_size=config["data"]["alphabet_size"],
                         embedding_size=config["char_cnn_zhang"]["embedding_size"],
                         conv_layers=config["char_cnn_zhang"]["conv_layers"],
                         fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                         num_of_classes=config["data"]["num_of_classes"],
                         threshold=config["char_cnn_zhang"]["threshold"],
                         dropout_p=config["char_cnn_zhang"]["dropout_p"],
                         optimizer=config["char_cnn_zhang"]["optimizer"],
                         loss=config["char_cnn_zhang"]["loss"])
    # Train model
    print("Start Training")
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])

    print ("Start Testing")
    model.test(testing_inputs=validation_inputs,
               testing_labels=validation_labels,
               batch_size=64)



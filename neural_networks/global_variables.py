'''GENERAL VARIABLES'''
device = "cpu"


#region MNIST Questions HyperParameter Definitions and Declarations
# Hyper parameters Definition:
mnist_hidden_size = 100
mnist_num_of_classes = 10
mnist_num_of_epochs = 20
mnist_batch_size = 100
mnist_learning_rate = 1e-4
mnist_weight_decay = 1e-5

def get_minst_input_size(dataset):
    return len(dataset[0][1])
#endregion


#region Simple Feed Forward ( Q1,4) HyperParameter Definitions and Declarations
# Hyperparameters
insurability_batch_size = 1
insurability_epochs = 30
insurability_learning_rate = 1e-7
insurability_nodes_per_layer = 16
#endregion
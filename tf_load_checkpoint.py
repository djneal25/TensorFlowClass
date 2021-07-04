import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

tf.compat.v1.disable_eager_execution()

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.compat.v1.variable_scope('input'):
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.compat.v1.variable_scope('layer_1'):
    weights = tf.compat.v1.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes],
                                        initializer=tf.keras.initializers.glorot_normal())
    biases = tf.compat.v1.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.compat.v1.variable_scope('layer_2'):
    weights = tf.compat.v1.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes],
                                        initializer=tf.keras.initializers.glorot_normal())
    biases = tf.compat.v1.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.compat.v1.variable_scope('layer_3'):
    weights = tf.compat.v1.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes],
                                        initializer=tf.keras.initializers.glorot_normal())
    biases = tf.compat.v1.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.compat.v1.variable_scope('output'):
    weights = tf.compat.v1.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs],
                                        initializer=tf.keras.initializers.glorot_normal())
    biases = tf.compat.v1.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.compat.v1.variable_scope('cost'):
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.compat.v1.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.compat.v1.variable_scope('train'):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.compat.v1.variable_scope('logging'):
    tf.compat.v1.summary.scalar('current_cost', cost)
    summary = tf.compat.v1.summary.merge_all()

saver = tf.compat.v1.train.Saver()

# Initialize a session so that we can run TensorFlow operations
with tf.compat.v1.Session() as session:

    # When loading from a checkpoint, don't initialize the variables!
    # session.run(tf.compat.v1.global_variables_initializer())

    # Instead, load them from disk:
    saver.restore(session, "logs/trained_model.ckpt")

    print("Trained model loaded from disk.")
    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.compat.v1.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.compat.v1.summary.FileWriter("./logs/testing", session.graph)

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trainined, let's use it to make prediction for the total earnings
    # Pass in the X testing data and run the "prediction" operation
    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})

    # Unscale the data back to it's original units (dollars)
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    real_earnings = test_data_df['total_earnings'].values[0]
    predicted_earnings = Y_predicted[0][0]

    print("The actual earnings of Game #1 were ${}".format(real_earnings))
    print("Out neural network predicted earnings of ${}".format(predicted_earnings))

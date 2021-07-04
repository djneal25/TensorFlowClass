# Import `tensorflow`
import os

import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.disable_eager_execution()

# Initialize two constants
X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")

# Initialize Session and run `result`
with tf.compat.v1.Session() as session:
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})
    print(result)

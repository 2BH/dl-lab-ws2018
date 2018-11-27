import tensorflow as tf

class Model:
    def __init__(self, history_length=1, learning_rate=3e-4):
        
        # TODO: Define network
        self.learning_rate = learning_rate 
        # variable for input and labels
        self.x_input = tf.placeholder(dtype=tf.float32, shape = [None, 96, 96, history_length], name = "x_input")
        self.y_label = tf.placeholder(dtype=tf.float32, shape = [None, 3], name = "y_label")

        # first layers + relu
        self.W_conv1 = tf.get_variable("W_conv1", [8, 8, history_length, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(self.x_input, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID')
        conv1_a = tf.nn.relu(conv1)
        # second layer + relu: 
        self.W_conv2 = tf.get_variable("W_conv2", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(conv1_a, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        conv2_a = tf.nn.relu(conv2)
        # third layer + relu:
        self.W_conv3 = tf.get_variable("W_conv3", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(conv2_a, self.W_conv3, strides=[1, 2, 2, 1], padding='VALID')
        conv3_a = tf.nn.relu(conv3)
        # forth layer + relu:
        self.W_conv4 = tf.get_variable("W_conv4", [3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer()) 
        conv4 = tf.nn.conv2d(conv3_a, self.W_conv4, strides=[1, 2, 2, 1], padding='VALID')
        conv4_a = tf.nn.relu(conv4)

        flatten = tf.contrib.layers.flatten(conv4_a)
        # first dense layer + relu + dropout
        fc1 = tf.contrib.layers.fully_connected(flatten, 400, activation_fn=tf.nn.relu)
        fc1_drop = tf.nn.dropout(fc1, 0.7)
        # second dense layer + relu:
        fc2 = tf.contrib.layers.fully_connected(fc1_drop, 400, activation_fn=tf.nn.relu)
        fc2_drop = tf.nn.dropout(fc2, 0.7)
        # third dense layer + relu 
        fc3 = tf.contrib.layers.fully_connected(fc2_drop, 50, activation_fn=tf.nn.relu)
        # output layer:
        self.output = tf.contrib.layers.fully_connected(fc3, 3, activation_fn=None)
        # TODO: Loss and optimizer
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.output, labels=self.y_label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # TODO: Start tensorflow session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)

import tensorflow as tf

class Model:
    def __init__(self, history_length=1, learning_rate=3e-4):
        
        # TODO: Define network
        self.learning_rate = learning_rate 
        # variable for input and labels
        self.x_input = tf.placeholder(dtype=tf.float32, shape = [None, 96, 96, history_length], name = "x_input")
        self.y_label = tf.placeholder(dtype=tf.float32, shape = [None, 4], name = "y_label")

        # first layers + relu
        self.W_conv1 = tf.get_variable("W_conv1", [8, 8, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
        z1 = tf.nn.conv2d(self.x_input, self.W_conv1, strides=[1, 4, 4, 1], padding='VALID')
        a1 = tf.nn.relu(z1)
        # second layer + relu: 
        self.W_conv2 = tf.get_variable("W_conv2", [4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        z2 = tf.nn.conv2d(a1, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        a2 = tf.nn.relu(z2)
        # third layer + relu:
        self.W_conv3 = tf.get_variable("W_conv3", [3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer())
        z3 = tf.nn.conv2d(a2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        a3 = tf.nn.relu(z3)
        
        flatten = tf.contrib.layers.flatten(a3)
        # first dense layer + relu
        z4 = tf.contrib.layers.fully_connected(flatten, 128, activation_fn=tf.nn.relu)

        # second dense layer + relus:
        z5 = tf.contrib.layers.fully_connected(z4, 128, activation_fn=tf.nn.relu)

        # output layer:
        self.output = tf.contrib.layers.fully_connected(z5, 4, activation_fn=tf.nn.relu)

        # TODO: Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_label))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # TODO: Start tensorflow session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)

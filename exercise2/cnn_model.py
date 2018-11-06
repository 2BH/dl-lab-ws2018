import tensorflow as tf
import numpy as np

class LeNet_class:
# convolutional neural network
    
    # constructor
    def __init__(self, learning_rate = 0.01, num_filters = 16, batch_size = 64, num_epochs = 100, model_name = "lenet"):
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name = model_name

    # create placeholder for input data
    def create_placeholder(self, x_shape, y_shape):
        # variable for input and labels
        self.x_input = tf.placeholder(dtype=tf.float32, shape = [None, *x_shape, 1], name = "x_input")
        self.y_label = tf.placeholder(dtype=tf.float32, shape = [None, y_shape], name = "y_label")

    # init parameters for training the model
    def init_parameter(self, w_filter, h_filter):
        # first layer: convolution + relu + max_pooling
        self.W_conv1 = tf.get_variable("W_conv1", [h_filter, w_filter, 1, self.num_filters], initializer=tf.contrib.layers.xavier_initializer())
        # second layer: convolution + relu + max_pooling
        self.W_conv2 = tf.get_variable("W_conv2", [h_filter, w_filter, self.num_filters, self.num_filters], initializer=tf.contrib.layers.xavier_initializer())

    # forward passing
    def forward_propagation(self, x_input):
        # first layer: convolution + relu + max_pooling
        # convolution: strides = 1, padding = same
        z1 = tf.nn.conv2d(x_input, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        # relu
        g1 = tf.nn.relu(z1)
        # max_pooling: size 2x2, strides = 1, padding = same
        p1 = tf.nn.max_pool(g1, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding='SAME')
        
        # second layer: convolution + relu + max_pooling  
        # convolution: strides = 1, padding = same
        z2 = tf.nn.conv2d(p1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        # relu
        g2 = tf.nn.relu(z2)
        # max_pooling: size 2x2, strides = 1, padding = same
        p2 = tf.nn.max_pool(g2, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding='SAME')
        
        # third layer: fully connected layer
        # falttern
        flatten = tf.contrib.layers.flatten(p2)
        # fully connected layer with 128 units
        z3 = tf.contrib.layers.fully_connected(flatten, 128, activation_fn=None)

        # output layer:
        z4 = tf.contrib.layers.fully_connected(z3, 10, activation_fn=None)

        return z4

    # compute the cost
    def compute_cost(self, output):
        # softmax with cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.y_label))
        return cost

    # create saver
    #def create_saver(self):
    #    self.model_saver = tf.train.Saver()

    # save the session to given path
    def save_model(self, sess):
        self.model_saver = tf.train.Saver()
        path = "./model/" + self.model_name
        self.model_saver.save(sess, path)

    # load the sesson from given path
    def load_session(self, path):
        tf.reset_default_graph()
        sess = tf.Session()
        #self.

    # train the model
    def train(self, X_train, y_train, X_valid, y_valid):
        # reset the model
        tf.reset_default_graph()
        # init the model
        self.create_placeholder(X_train.shape[1:3], y_train.shape[1])
        self.init_parameter(3, 3)
        total_batch_num = X_train.shape[0] // self.batch_size
        train_cost = np.zeros((self.num_epochs))
        train_accuracy = np.zeros((self.num_epochs))
        valid_accuracy = np.zeros((self.num_epochs))

        # build up compute graph
        output = self.forward_propagation(self.x_input)
        cost = self.compute_cost(output)

        # define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # initialize variables at the same time
        init = tf.global_variables_initializer()

        # calculate the accuracy
        y_pred = tf.argmax(output, 1)
        tf.add_to_collection('pred_network', y_pred)
        correct_pred = tf.equal(y_pred, tf.argmax(self.y_label, 1))
        # calculate train and valid accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
        #tf.add_to_collection('accuracy', accuracy)
        # initialization
        with tf.Session() as sess:
            sess.run(init)
            # training the model with mini batch
            for epoch in range(self.num_epochs):
                for b in range(total_batch_num):
                    # select the batch data
                    X_batch = X_train[b*self.batch_size:(b+1)*self.batch_size,:,:,:]
                    y_batch = y_train[b*self.batch_size:(b+1)*self.batch_size,:]
                        # compute the cost
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={self.x_input:X_batch, self.y_label:y_batch})
                    train_cost[epoch] += temp_cost / self.batch_size

                train_accuracy[epoch] = accuracy.eval({self.x_input: X_train, self.y_label: y_train})
                valid_accuracy[epoch] = accuracy.eval({self.x_input: X_valid, self.y_label: y_valid})
                print("[%d/%d]: train_accuracy: %.4f, valid_accuracy: %.4f" %(epoch+1, self.num_epochs, train_accuracy[epoch], valid_accuracy[epoch]))

            # save the model
            self.save_model(sess)

        return train_accuracy, valid_accuracy, train_cost

    # evaluate the model
    def eval(self, X_test, y_test):
        if not tf.contrib.framework.is_tensor(self.x_input):
            raise NotTrainedError("This model has not trained.")

        filepath = "./model/" + self.model_name + ".meta"
        saver = tf.train.import_meta_graph(filepath)
        with tf.Session() as sess:
            # load the session
            saver.restore(sess, tf.train.latest_checkpoint("./model"))
            # calculate the test error
            y_pred = tf.get_collection("pred_network")[0]
            prediction = np.array(sess.run(y_pred, feed_dict = {self.x_input: X_test}))
            test_error = float(np.sum(prediction != np.argmax(y_test, axis = 1)) / y_test.shape[0])
            print("test error: %.4f" %test_error)

        return test_error



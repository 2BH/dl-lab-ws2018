from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation
import tensorflow as tf

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def data_augmentation(X_train, y_train_id):
    # flip LEFT and RIGHT actions
    # CAN ONLY use when history_length = 0
    left_indices = (y_train_id == LEFT)
    right_indices = (y_train_id == RIGHT)
    X_new_left_data = np.flip(X_train[left_indices], axis=2)
    X_new_right_data = np.flip(X_train[right_indices], axis=2)
    y_new_left_data = np.zeros((X_new_left_data.shape[0])) + RIGHT
    y_new_right_data = np.zeros((X_new_right_data.shape[0])) + LEFT
    X_train_n = np.concatenate((X_train, X_new_left_data, X_new_right_data), axis=0)
    y_train_id_n = np.concatenate((y_train_id, y_new_left_data, y_new_right_data), axis=0)
    
    return X_train_n, y_train_id_n

def id_to_action(labels_id):
    # convert id format to action format
    classes = 3
    labels_action = np.zeros((labels_id.shape[0], classes))
    labels_action[labels_id==LEFT] = [-1.0, 0.0, 0.0]
    labels_action[labels_id==RIGHT] = [1.0, 0.0, 0.0]
    labels_action[labels_id==STRAIGHT] = [0.0, 0.0, 0.0]
    labels_action[labels_id==ACCELERATE] = [0.0, 1.0, 0.0]
    labels_action[labels_id==BRAKE] = [0.0, 0.0, 0.2]
    labels_action[labels_id==LEFT_BRAKE] = [-1.0, 0.0, 0.2]
    labels_action[labels_id==RIGHT_BRAKE] = [1.0, 0.0, 0.2]
    labels_action[labels_id==LEFT_ACCELERATE] = [-1.0, 1.0, 0.0]
    labels_action[labels_id==RIGHT_ACCELERATE] = [1.0, 1.0, 0.0]
    
    return labels_action

def uniform_sampling(X_train, y_train_id_n, num_samples):
    n = X_train.shape[0]
    weights = np.zeros(n)
    left_indices = y_train_id_n == LEFT
    weights[left_indices] = n / np.sum(left_indices)
    right_indices = y_train_id_n == RIGHT
    weights[right_indices] = n / np.sum(right_indices)
    straight_indices = y_train_id_n == STRAIGHT
    weights[straight_indices] = n / np.sum(straight_indices)
    acce_indices = y_train_id_n == ACCELERATE
    weights[acce_indices] = n / np.sum(acce_indices)
    brake_indices = y_train_id_n == BRAKE
    weights[brake_indices] = n / np.sum(brake_indices)
    left_brake_indices = y_train_id_n == LEFT_BRAKE
    weights[left_brake_indices] = n / np.sum(left_brake_indices)
    right_brake_indices = y_train_id_n == RIGHT_BRAKE
    weights[right_brake_indices] = n / np.sum(right_brake_indices)
    left_acce_indices = y_train_id_n == LEFT_ACCELERATE
    weights[left_acce_indices] = n / np.sum(left_acce_indices)
    right_acce_indices = y_train_id_n == RIGHT_ACCELERATE
    weights[right_acce_indices] = n / np.sum(right_acce_indices)
    weights = weights / np.sum(weights)
    samples_indices = np.random.choice(np.arange(n), num_samples,replace=False,  p = weights)
    
    return samples_indices


def sample_minibatch(X, y, batch_index, history_length=1):
    # get small batch
    batch_size = batch_index.shape[0]
    X_batch = np.zeros((batch_size, X.shape[1], X.shape[2], history_length))
    for i in range(history_length):
        X_batch[:,:,:,i] = X[batch_index+i]
    y_batch = y[batch_index+history_length-1]
    return X_batch, y_batch


def preprocessing(X_train, y_train, X_valid, y_valid):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # convert to gray channel
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    n = X_train.shape[0]
    # discretize training set
    y_train_id = np.zeros(n)
    for i in range(n):
        y_train_id[i] = action_to_id(y_train[i])

    return X_train, y_train_id, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, epochs=10, batch_size=64, lr=0.0003, history_length=1, num_uniform_sample=12000, num_filters=64, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(learning_rate=lr, history_length=history_length, num_filters=num_filters)
    init = tf.global_variables_initializer()
    agent.sess.run(init)
    tensorboard_eval = Evaluation(tensorboard_dir)
    tf.reset_default_graph()

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    offset = history_length - 1

    # training loop
    # init training cost
    train_cost = np.zeros((epochs))
    valid_cost = np.zeros((epochs))
    for epoch in range(epochs):
        index = uniform_sampling(X_train[offset:], y_train[offset:], num_uniform_sample)
        total_batch_num = (num_uniform_sample - history_length + 1) // batch_size;
        total_batch_num_valid = (X_valid.shape[0] - history_length + 1)// batch_size;
        
        for b in range(total_batch_num):
            # select the batch data
            batch_index = index[b*batch_size:(b+1)*batch_size]
            X_batch, y_batch = sample_minibatch(X_train, y_train, batch_index, history_length)
            y_batch = id_to_action(y_batch)
            # compute the cost
            _ , temp_cost = agent.sess.run([agent.optimizer, agent.cost], feed_dict={agent.x_input:X_batch, agent.y_label:y_batch})

        # training cost
        for b in range(total_batch_num):
            batch_index = index[b*batch_size:(b+1)*batch_size]
            X_batch, y_batch = sample_minibatch(X_train, y_train, batch_index, history_length)
            y_batch = id_to_action(y_batch)
            train_cost[epoch] += agent.sess.run(agent.cost, feed_dict={agent.x_input: X_batch, agent.y_label: y_batch})

        # validation cost
        for b in range(total_batch_num_valid):
            batch_index = np.arange(b*batch_size,(b+1)*batch_size)
            X_valid_batch, y_valid_batch = sample_minibatch(X_valid, y_valid, batch_index, history_length)
            valid_cost[epoch] += agent.sess.run(agent.cost, feed_dict={agent.x_input:X_valid_batch, agent.y_label:y_valid_batch})
        train_cost[epoch] = train_cost[epoch] / total_batch_num
        valid_cost[epoch] = valid_cost[epoch] / total_batch_num_valid
        print("[%d/%d]: train_cost: %.4f, valid_cost: %.4f" %(epoch+1, epochs, train_cost[epoch], valid_cost[epoch]))
        eval_dict = {"train":train_cost[epoch], "valid":valid_cost[epoch]}
        tensorboard_eval.write_episode_data(epoch, eval_dict)
      
    # TODO: save your agent
    agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)
    agent.sess.close()
    return train_cost, valid_cost

if __name__ == "__main__":
    # best hypeparameters with lstm:
    #  {'num_filters': 41,
    # 'learning_rate':0.0001339095972726421,
    # 'batch_size': 28,
    # 'history_length': 2,
    # 'num_uniform_sample': 16000}

    # best hyperparameters without lstm:
    # num_filters : 74
    # batch_size: 58
    # history_length: 1
    # num_uniform_sample: 16000
    # learning_rate: 0.00032224967019634816

    # hyperparameters
    plot = True
    used_num_samples = 30000
    history_length = 2
    epochs = 20
    batch_size = 28
    learning_rate = 0.000133905972726421
    num_filters = 41
    # each epoch use how many samples
    num_uniform_sample = 16000
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    start = X_train.shape[0] - used_num_samples
    X_train = X_train[start:]
    y_train = y_train[start:]

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid)
    # train model (you can change the parameters!)
    train_cost, valid_cost = train_model(X_train, y_train, X_valid, y_valid, history_length=history_length, epochs=epochs, batch_size=batch_size, lr=learning_rate, num_uniform_sample=num_uniform_sample, num_filters=num_filters)
    
    if plot == True:
        plt.plot(np.arange(epochs)+1, train_cost, label="training cost")
        plt.plot(np.arange(epochs)+1, valid_cost, label="validation cost")
        plt.legend()
        plt.title("train/valid cost wrt training epochs")
        plt.xlabel("# epoch")
        plt.ylabel("cost")
        plt.show()
        plt.savefig("./train-valid-cost.png")
 

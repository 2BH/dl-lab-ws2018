import numpy as np
LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))


def id_to_action(labels_id):
    # convert id format to action format
    classes = 3
    labels_action = np.zeros(classes)
    labels_action[labels_id==1] = [-1.0, 0.0, 0.0]
    labels_action[labels_id==2] = [1.0, 0.0, 0.0]
    labels_action[labels_id==0] = [0.0, 0.0, 0.0]
    labels_action[labels_id==3] = [0.0, 1.0, 0.0]
    labels_action[labels_id==4] = [0.0, 0.0, 0.8]
    
    return labels_action

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 
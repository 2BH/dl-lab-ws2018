import gym

env = gym.make('MsPacman-v0')

print(env.env.get_action_meanings())
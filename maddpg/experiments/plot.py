import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

import pickle


def parse_args():
    parser = argparse.ArgumentParser("Plot Results")
    # Environment
    parser.add_argument("--path", type=str, default="./learning_curves/formation1_rewards.pkl", help="path of saving data for plotting")
    parser.add_argument("--rewards", type=str, default="complete", help="path of saving data for plotting")

    return parser.parse_args()


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def plot(arglist):
    file = open(arglist.path, "rb")
    data = pickle.load(file)

    if arglist.rewards == "agent":  
        # for agent_reward in data:
        #     agent_reward = smooth(agent_reward, 5)
        #     plt.plot(range(1, len(agent_reward) + 1), agent_reward)
        agent_reward0 = data[0]
        agent_reward1 = data[1]
        agent_reward2 = data[2]
        agent_reward0 = smooth(agent_reward0, 10)
        agent_reward1 = smooth(agent_reward1, 10)
        agent_reward2 = smooth(agent_reward2, 10)
        plt.plot(range(1, len(agent_reward0) + 1), agent_reward0)
        plt.plot(range(1, len(agent_reward1) + 1), agent_reward1)
        plt.plot(range(1, len(agent_reward2) + 1), agent_reward2, linestyle = '--')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        print(agent_reward0)
        print(agent_reward1)
        print(agent_reward2)

    if arglist.rewards == "complete":
        data = smooth(data, 10)
        plt.plot(range(1, len(data) + 1), data)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        print(data)

    file.close()


if __name__ == '__main__':
    arglist = parse_args()
    plot(arglist)

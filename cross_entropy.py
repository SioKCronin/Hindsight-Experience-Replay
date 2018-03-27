# Cross Entropy

import numpy as np
import gym
import gym.spaces import Discrete, Box

class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        dim_ob = ob_space.shape[0] # dimension of observations
        n_actions = ac_space.n # number of actions
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew

env = None
def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta,
                env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta,
                env.observation_space, env.action_space)
    else:
        raise NotImplementedError

# Task settings
env = gym.make('CartPole-v0')
num_steps = 500

# Algorithm settings
n_iter = 100
batch_size = 25
elite_frac = 0.2

if isinstance(env.action_space, Discrete):
    dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = (env.observation_space_shape[0]+1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# Initialize mean and std dev
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

# CEM!
for iteration in range(n_iter):
    # Sample paramater vectors
    thetas = [random init randoms of size theta]
    rewards = [noisy_evaluation(theta) for theta in thetas]

    # Get elite params
    n_elite = int(batch_size * elite_frac)
    elite_inds = np.argsort(rewards)[batch_size - n_elite: batch_size]
    elite_thetas = [thetas[i] for i in elite_inds]

    # Update theta_mean, theta_std
    theta_mean = calc mean of elites
    theta_std = calc std dev of elites


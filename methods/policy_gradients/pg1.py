# Policy Gradients

'''
Stochastic policy that samples actions, and then actions that happy to
eventually lead to good outcomes get encouraged in the future,
and actions taken that lead to bad outcomes get discouraged.
'''

import numpy as np
import gym
import pickle

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2 ???
resume = False # resume from previous checkpoint
render = False

# initialize model
D = 80 * 80 # grid
if resume: 
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # 'Xavier' initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k : np.zeros_like(v) for k,v in model.items()} # update buffers that add up gradients over a batch
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.items()} #rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid squashing to [0, 1]


def prepro(I):
    """prepro 210x160x3 unit8 frame into 6400 (80 x 80) 1D float vector"""

    I = I[35:195] # crop
    I = I[::2,::2,0] #downsample by factor of 2
    I[I ==144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (type 2)
    I[I != 0] = 1 # set everything else to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # it's a pong thing - reset the sum for game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return prob of taking action 2, and hideen state

def policy_backward(eph, epdlogp):
    """backward pass (eph is array of intermediate hidden states)"""
    dW2 = np.dot(eph.T, epdlogp). ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 #backprop prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used to compute difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    # preprocess the observation, set network input to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #forward the policy network and sample an action from the returned prob
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    #record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # encourages the action that was taken to be taken

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (called after step to reward for prev action)

    if done: # an episode finished
        episode_number += 1

        #stack together all inputs, hidden states, action gradients, and reward for episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], [] # reset array memory

        # compute discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG!)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] #accumulate grad over batch

        # perform rmsprop parm update after every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # Keep track of what's happening
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward when game ends
        print (('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !! Woot woot!!!'))

    





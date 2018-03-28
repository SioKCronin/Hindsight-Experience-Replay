# Policy Gradients 2 (after John Schulman)

import numpy as np
import os
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64"
import theano, theano.tensor as T
import gym

def discount(x, gamma):


def categorical_sample(prob_n):


def get_traj(agent, env, episode_max_length, render=False):


def sgd_updates(grads, params, stepsize):


def rmsprop_updates(grads, params, stepsize, rho=0.9, epsilon=1e-9):


class REINFORCEAgent(object):

    def __init__(self, ob_space, action_space, **usercfg):
        nO = ob_space.shape[0]
        nA = action_space.n

        self.config = dict(episode_max_length=100, timesteps_per_batch=10000,
                n_iter=100, gamma=1.0, stepsize=0.05, nhid=20)
        self.config.update(usercfg)

        # Variables for observation, action, and advantage
        ob_no = T.fmatrix() # Observation
        a_n = T.ivector() # Discrete action
        adv_n = T.fvector() # Advantage
        def shared(arr):
             return theano.shared(arr.astype('float64'))
        
        # Create weights of neural network with one hidden layer
        W0 = shared(np.random.randn(nO, self.config['nhid'])/np.sqrt(nO))
        b0 = shared(np.zeros(self.config['nhid']))
        W1 = shared(1e-4*np.random.randn(self.config['nhid'],nA))
        b1 = shared(np.zeros(nA))
        params = [W0, b0, W1, b1]
        
        # Action probs
        prob_na = T.nnet.softmax(T.tanh(ob_no.dot(W0)+b0[None,:]).dot(W1)+b1[None,:])
        N = ob_no.shape[0]

        # Loss function we differentiate to get the policy gradient
        # (divided by total number of timesteps)
        loss = T.log(prob_na[T.arrange(N), a_n]).dot(adv_n)/N
        stepsize = T.fscaler()
        grads = T.grad(loss, params)

        # Perform param updates
        updates = rmpsprop_updates(grads, params, stepsize)
        self.pg_update = theano.function([ob_no, a_n, adv_n, stepsize],[], updates=updates, allow_input_downcast=True)
        self.compute_prob = theano.function([ob_no], prob_na, allow_input_downcast=True)

def act(self, ob):
    prob = self.compute_prob(ob.reshape(1,-1))
    action = categorical_sample(prob)
    return action

def learn(self, env):
    cfg = self.config
    for iteration in range(cfg["n_iter"]):
        trajs = []
        timesteps_total = 0
        while timesteps_total < cfg["timesteps_per_batch"]:
            traj = get_traj(self, env, cfg["episode_max_length"])
            trajs.append(traj)
            timesteps_total += len(traj["reward"])
        all_ob = np.concatenate([traj["ob"] for traj in trajs])

        # Compute discounted sums of rewards
        rets = [discount(traj["reward"], cfg["gamma"]) for traj in trajs]
        maxlen = max(len(ret) for ret in rets)
        padded_rets = [np.concatenate([ret, np.zeros(maxlen-len(ret))]) for ret in rets]

        # Compute time-dependent baseline
        baseline = np.mean(padded_rets, axis=0)

        # Compute advantage function
        advs = [ret - baseline[:len(ret)] for ret in rets]
        all_action = np.concatenate([traj["action"] for traj in trajs])
        all_adv = np.concatenate(advs)

        # Policy gradient update step
        self.pg_update(all_ob, all_action, all_adv, cfg["stepsize"])
        eprews = np.array([traj["reward"].sum() for traj in trajs]) # episode total rewards
        eplens = np.array([len(traj["rewards"]) for traj in trajs]) # episode lengths

        # Print out stats
        print("Iterations: \t %i" %iterations)
        # more

def main():
    env = gym.make("Acrobot-v0")
    agent = REINFORCEAgent(env.observation_space, env.action_space,
            episode_max_length=env.spec.timestep_limit)
    agent.learn(env)

if __name__ == "__main__":
    main()








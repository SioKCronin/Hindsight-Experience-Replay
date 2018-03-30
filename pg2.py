# Policy Gradients 2 (after John Schulman)

import numpy as np
import os
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64"
import theano
import theano.tensor as T
import tensorflow as tf
import gym

def discount(x, gamma):
    '''
    Given a vector x, computes a vector y such that
    y[i] = x[i] + gamma*x[i+1] + gamma^2*x[i+2] + ...
    We care less about events far in the future
    '''
    out = np.zeros(len(x), 'float64')
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >=1
    return out

def categorical_sample(prob_n):
    '''
    Sample from categorical distribution, specified by a vector of class probs.
    '''
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def get_traj(agent, env, episode_max_length, render=False):
    '''
    Run agent-env loop for one whole episode (trajectory)
    Return dict of results
    '''
    ob = env.reset()
    obs = []
    acts = []
    rews = []
    for _ in range(episode_max_length):
        a = agent.act(ob)
        (ob, rew, done, _) = env.step(a)
        obs.append(ob)
        acts.append(a)
        rews.append(rew)
        if done: break
        if render: env.render()
    return {"reward" : np.array(rews),
            "ob": np.array(obs),
            "action": np.array(acts)
            }

#def sgd_updates(grads, params, stepsize):

def rmsprop_updates(grads, params, stepsize, rho=0.9, epsilon=1e-9):
    updates = []

    for param, grad in zip(params, grads):
        accum = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=param.dtype))
        accum_new = rho * accum + (1 - rho) * grad**2
        updates.append((accum, accum_new))
        updates.append((param, param + (stepsize * grad / T.sqrt(accum_new + epsilon))))
    return updates

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

        # Calculate alternate action vector of most likely action
        print("prob_na", prob_na[T.arange(N), a_n])

        # Loss function we differentiate to get the policy gradient
        # (divided by total number of timesteps)
        loss = T.log(prob_na[T.arange(N), a_n]).dot(adv_n)/N
        stepsize = T.fscalar()
        grads = T.grad(loss, params)

        # Perform param updates
        updates = rmsprop_updates(grads, params, stepsize)
        self.pg_update = theano.function([ob_no, a_n, adv_n, stepsize],[],
                updates=updates, allow_input_downcast=True)
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
            eplens = np.array([len(traj["reward"]) for traj in trajs]) # episode lengths

            # Print out stats
            print("----------------------")
            print("Iterations: \t %i" %iteration)
            print("NumTrajs: \t %i" %len(eprews))
            print("NumTimesteps: \t %i" %np.sum(eplens))
            print("MaxRew: \t %s" %eprews.max())
            print("MeanRew: \t %s +- %s" % (eprews.mean(),
                eprews.std()/np.sqrt(len(eprews))))
            print("MeanLen: \t %s +- %s" % (eplens.mean(),
                eplens.std()/np.sqrt(len(eplens))))
            print("----------------------")
            get_traj(self, env, cfg["episode_max_length"], render=True)

def main():
    env = gym.make("Acrobot-v1")
    agent = REINFORCEAgent(env.observation_space, env.action_space,
            episode_max_length=env.spec.timestep_limit)
    agent.learn(env)

if __name__ == "__main__":
    main()








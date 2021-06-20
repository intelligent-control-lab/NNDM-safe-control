import os
import numpy as np
import gym
import pandas as pd
import progressbar
import gym_dynamics
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('error', RuntimeWarning)

class SafeLearning(object):
    def __init__(self, CMAES_args):
        """
        ================================================================================
        Initialize the learning module. Create learning log.
        """
        
        self.cmaes_args = CMAES_args

        now = datetime.now()

        timestamp = now.strftime("%m-%d_%H:%M")
        log_name = CMAES_args["exp_prefix"] + "_epoch:" + str(CMAES_args["epoch"]) + "_populate_num:" + str(CMAES_args["populate_num"]) + "_elite_ratio:" + str(CMAES_args["elite_ratio"]) + "_init_sigma_ratio:" + str(CMAES_args["init_sigma_ratio"]) + "_noise_ratio:" + str(CMAES_args["noise_ratio"]) + "_date:" + timestamp 
        self.log = open(os.path.dirname(os.path.abspath(__file__)) + "/../data/learning_log/" + log_name + ".txt","w")

        # self.evaluator = self.evaluate_insertion
        self.evaluator = self.cmaes_args["evaluator"]
    
    def regulate_params(self, params):
        """
        ================================================================================
        Regulate params by upper bound and lower bound. And convert params to integer if required by the user.
        """
        params = np.maximum(params, self.cmaes_args["lower_bound"]) # lower bound
        params = np.minimum(params, self.cmaes_args["upper_bound"]) # upper bound
        if "param_is_int" in self.cmaes_args:
            for i in range(params.shape[1]):
                if self.cmaes_args["param_is_int"][i]:
                    params[:,[i]] = np.vstack([int(round(x)) for x in params[:,i]])
            # params = [ if self.cmaes_args["param_is_int"][i] else params[i] ]
        return params

    def populate(self, mu, sigma):
        """
        ================================================================================
        Populate n members using the current estimates of mu and S
        """
        self.population = np.random.multivariate_normal(mu, sigma, self.cmaes_args["populate_num"])
        self.population = self.regulate_params(self.population)

    def evaluate(self, mu, log=True):
        """
        ===============================================================================
        Evaluate a set of weights (a mu) by interacting with the environment and
        return the average total reward over multiple repeats.
        """
        
        print("\n---------------------------\nevaluating mu = {}".format(mu))
        self.evaluator.set_params(mu)

        rewards = []
        repeat_times = 1  # test multiple times to reduce randomness
        for i in range(repeat_times):
            # reward = self.evaluate_insertion(mu)
            reward = self.evaluator.evaluate(mu)
            rewards.append(reward)

        print('Rewards: {}'.format(rewards))

        reward = np.mean(rewards)
        if log:
            self.log.write("{} {}".format(str(mu), reward))
            self.log.write(self.evaluator.log+"\n")
            self.log.flush()
        return reward

    def step(self, mu, sigma):
        """
        ===============================================================================
        Perform an iteration of CMA-ES by evaluating all members of the current 
        population and then updating mu and S with the top self.cmaes_args["elite_ratio"] proportion of members 
        and updateing the weights of the policy networks.
        """
        self.populate(mu, sigma)
        rewards = np.array(list(map(self.evaluate, self.population)))
        indexes = np.argsort(-rewards) 
        """
        ===============================================================================
        best members are the top self.cmaes_args["elite_ratio"] proportion of members with the highest 
        evaluation rewards.
        """
        best_members = self.population[indexes[0:int(self.cmaes_args["elite_ratio"] * self.cmaes_args["populate_num"])]]
        mu = np.mean(best_members, axis=0)
        sigma = np.cov(best_members.T) + self.noise
        print("avg best mu in this epoch:")
        print(mu)
        return mu, sigma

    def learn(self):
        mu = self.cmaes_args["init_params"]
        bound_range = np.array(self.cmaes_args["upper_bound"]) - np.array(self.cmaes_args["lower_bound"])
        sigma = np.diag((self.cmaes_args["init_sigma_ratio"] * bound_range)**2)
        self.noise = np.diag((self.cmaes_args["noise_ratio"] * bound_range)**2)
        
        for i in range(self.cmaes_args["epoch"]):
            
            self.log.write("epoch {}\n".format(i))
            mu, sigma = self.step(mu, sigma)
            print("learning")

        print("Final best param:")
        print(mu)
        print("Final reward:")
        print(self.evaluate(mu, log=False))
        return mu


class UnicycleSafetyIndex():
    def __init__(self):
        self.coe = np.zeros(5)
        nx = 19
        ny = 19
        nv = 9
        nt = 9
        self.xs = np.linspace(0, 5, nx+1)
        self.ys = np.linspace(0, 5, ny+1)
        self.vs = np.linspace(-2, 2, nv+1)
        self.ts = np.linspace(-np.pi, np.pi, nt+1)
        xm, ym, vm, tm = np.meshgrid(self.xs, self.ys, self.vs, self.ts)

        self.samples = []
        for i,j,k,l in np.ndindex(xm.shape):
            # self.samples.append(([xm[i,j,k,l], ym[i,j,k,l], vm[i,j,k,l], tm[i,j,k,l]], [0,0]))
            self.samples.append(([i,j,k,l], [0,0]))

        env = gym.make("Unicycle-v0")
        self.max_u = env.max_u
        self.dt = env.dt

        self.d_min = 1

    def phi(self, x, o):
        d = (x[0]-o[0])**2 + (x[1]-o[1])**2
        vx = x[2] * np.cos(x[3])
        vy = x[2] * np.sin(x[3])
        dot_d = 2*(x[0]-o[0])*vx + 2*(x[1]-o[1])*vy
        return self.d_min - d**self.coe[1] - self.coe[2] * dot_d

    def grad_phi(self, x, o):
        d = (x[0]-o[0])**2 + (x[1]-o[1])**2
        vx = x[2] * np.cos(x[3])
        vy = x[2] * np.sin(x[3])
        dot_d = 2*(x[0]-o[0])*vx + 2*(x[1]-o[1])*vy
        grad_d = np.array([2*(x[0]-o[0]), 2*(x[1]-o[1]), 0, 0])
        grad_dot_d = np.array([2*vx, 2*vy, 2*(x[0]-o[0])*np.cos(x[3]) + 2*(x[1]-o[1])*np.sin(x[3]), 2*(x[0]-o[0])*x[2]*(-np.sin(x[3])) + 2*(x[1]-o[1])*x[2]*(np.cos(x[3]))])
        # print(self.coe[1]-1)
        try:
            a = self.coe[1]*d**(self.coe[1]-1)*grad_d - self.coe[2]*grad_dot_d
        except:
            print(self.coe)
            print(d)
            print(grad_d)
            print(grad_dot_d)
            print(d**(self.coe[1]-1))
        return self.coe[1]*d**(self.coe[1]-1)*grad_d - self.coe[2]*grad_dot_d

    def set_params(self, params):
        self.coe = params

    def has_valid_control(self, C, d, x):
        f = np.vstack([x[2]*np.cos(x[3]), x[2]*np.sin(x[3]), 0, 0])
        g = np.vstack((np.zeros((2,2)), np.eye(2)))
        self.max_u
        return (C @ g @ self.max_u < d - C @ f) or (-C @ g @ self.max_u < d - C @ f)

    def evaluate(self, params):
        valid = 0
        for sample in self.samples:
            x_idx, o = sample
            x = [self.xs[x_idx[0]], self.ys[x_idx[1]], self.vs[x_idx[2]], self.ts[x_idx[3]]]
            phi = self.phi(x, o)
            # C dot_x < d
            # phi(x_k) > 0 -> con: dot_phi(x_k) < -k*phi(x): C = self.grad_phi(x, o)  d = -phi/dt*self.coe[0]
            # phi(x_k) < 0 -> con: phi(x_k+1) < 0:           C = self.grad_phi(x, o)  d = -phi/dt
            C = self.grad_phi(x, o)
            d = -phi/self.dt if phi < 0 else -phi/self.dt*self.coe[0]
            valid += self.has_valid_control(C, d, x)

        self.valid = valid
        return valid

    def visualize(self):
        valid_cnt = np.zeros((len(self.xs), len(self.ys)))
        tot_cnt = 0
        for sample in self.samples:
            x_idx, o = sample
            x = [self.xs[x_idx[0]], self.ys[x_idx[1]], self.vs[x_idx[2]], self.ts[x_idx[3]]]
            # print(x_idx)
            # print(x)
            phi = self.phi(x, o)
            C = self.grad_phi(x, o)
            d = -phi/self.dt if phi < 0 else -phi/self.dt*self.coe[0]
            if not self.has_valid_control(C, d, x):
                continue
            px,py,v,theta = x
            # if (px,py) not in valid_cnt.keys():
            #     valid_cnt[x_idx[0], x_idx[1]] = 0
            valid_cnt[x_idx[0], x_idx[1]] += 1
            tot_cnt += 1

        print("tot_cnt: ", tot_cnt)
        # plt.figure()
        sns.set_theme()
        ax = sns.heatmap(valid_cnt, cmap="YlGnBu", vmin=0)
        xticks = range(0, len(self.xs), len(self.xs)//5)
        yticks = range(0, len(self.ys), len(self.ys)//5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(np.array(self.xs[xticks]).astype(int))
        ax.set_yticklabels(np.array(self.ys[yticks]).astype(int))
        plt.ylim(0,len(self.ys))
        plt.xlim(0,len(self.xs))
        # plt.show()
        plt.savefig("../results/"+str(self.coe)+".png", dpi=300)
        # for key in valid_cnt.keys():
        #     # print(key)
        #     # print(valid_cnt[key])
        #     plt.scatter(key[0], key[1], color="b")
        # plt.show()

        # print(valid_cnt)
        return valid_cnt

    @property
    def log(self):
        return "{} {}".format(str(self.coe), str(self.valid))

def main():

    config = {
        "exp_prefix": "pm_learning",
        "epoch": 3,
        "elite_ratio": 0.06,
        "populate_num": 50,
        "init_params": [4, 2, 0.5],
        "lower_bound": [0, 1, 0],
        "upper_bound": [10, 4, 10],
        "init_sigma_ratio": 0.3,
        "noise_ratio": 0.01,
        "evaluator": UnicycleSafetyIndex(),
    }

    # si = UnicycleSafetyIndex()
    # si.set_params([5, 2, 0.5])
    # si.set_params([2, 2, 2])
    # si.visualize()

    # learner = SafeLearning(config)
    # learner.learn()

    si = UnicycleSafetyIndex()
    # si.set_params([1, 3, 2.8])
    # si.set_params([4, 2, 0.5])
    si.set_params([1, 1, 1])
    # si.set_params([0.,         1.0864768,  0.81123281])
    si.visualize()

if __name__ == "__main__":
    main()
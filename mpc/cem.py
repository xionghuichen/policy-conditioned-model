import numpy as np
import scipy.stats as stats


class CEMOptimizer(object):

    def __init__(self, planning_horizon, action_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25, num_particals=20):
        """Creates an instance of this class.

        Arguments:
            planning horizon (int)
            action dim (int)
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.planning_horizon = planning_horizon
        self.action_dim = action_dim
        self.sol_dim = planning_horizon * action_dim
        self.max_iters, self.popsize, self.num_elites = max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function
        self.num_particals = num_particals

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, current_obs):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        info = []

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            costs = self.cost_function(current_obs, samples.reshape(self.popsize, self.planning_horizon, self.action_dim), self.num_particals)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

            # action_seq = samples.reshape(self.popsize, self.planning_horizon, self.action_dim)
            # info.append({"obs": current_obs, "action_seq": action_seq, "costs": -costs})

        # np.save('pam_info.npy', info, allow_pickle=True)
        
        # print(f"cem iters: {t}, elites reward: {-np.sort(costs)[:self.num_elites].mean()}")

        return mean
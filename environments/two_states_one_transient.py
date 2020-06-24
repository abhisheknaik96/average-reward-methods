from environments.base_environment import OneHotEnv
import numpy as np


class TwoStatesTransientMDP(OneHotEnv):
    """
    Implements a simple MDP which will cause divergence in RVI-Q learning (Abounadi et al. 2001)
    """

    def __init__(self):
        super().__init__()

        self.num_states = 2
        self.num_actions = 2

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        self.P[0, 0, 0] = 0.9
        self.R[0, 0, 0] = 1
        self.P[0, 0, 1] = 0.1
        self.P[0, 1, 1] = 1
        self.R[0, 1, 1] = -10
        self.P[1, 0, 1] = 1
        self.P[1, 1, 1] = 1
        self.R[1, 0, 1] = 2
        self.R[1, 1, 1] = 2

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = 0    # the left state which is non-recurrent under all policies
        self.reward_obs_term = [0.0, None, False]


def main():

    env = TwoStatesTransientMDP()
    env.env_init({'random_seed': 0})
    ## test the transition and reward matrices
    print(env.P)
    print(env.R)

    obs = env.env_start()
    action = np.random.choice(env.num_actions)
    print(obs, action)
    ## test some observations and rewards
    for i in range(10):
        obs = env.env_step(action)
        print(obs[0], obs[1])
        action = np.random.choice(env.num_actions)
        print(action)


# to run this test, comment out all of the environments/__init__ file.
# the contents of that file are necessary for all the environments to be available in the outer folder's run_exp
if __name__ == '__main__':
    main()

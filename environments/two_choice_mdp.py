from environments.base_environment import OneHotEnv
import numpy as np

class TwoChoiceMDP(OneHotEnv):
    """
    Implements a variant of the environment from Mahadevan (1994),
    also used in this form by Naik et al. (2019) (https://arxiv.org/abs/1910.02140).
    """

    def __init__(self):
        super().__init__()
        self.num_states = 9
        self.num_actions = 2

        self.start_state = 0
        self.current_state = 0
        self.reward_scale_factor = None

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.reward_scale_factor = env_info.get('reward_scale_factor', 1)

        # adding connections from node i to i+1
        for s in range(self.num_states - 1):
            self.P[s, 0, s + 1] = 1
            self.P[s, 1, s + 1] = 1
        # connection from N-1th to 0th node
        self.P[8, 0, 0] = 1; self.P[8, 1, 0] = 1
        # removing the connection from 4th to 5th node
        self.P[4, 0, 5] = 0; self.P[4, 1, 5] = 0
        # connection from 4th to 0th node
        self.P[4, 0, 0] = 1; self.P[4, 1, 0] = 1
        # action 1 in node 0 should not lead to 1, but 5
        self.P[0, 1, 1] = 0
        self.P[0, 1, 5] = 1
        # rewards for going from 0 to 1 and 8 to 0
        self.R[0, 0, 1] = 1
        self.R[8, 0, 0] = 2; self.R[8, 1, 0] = 2
        self.R *= self.reward_scale_factor

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]


def main():

    env = TwoChoiceMDP()
    env.env_init()
    ## test the transition and reward matrices
    # print(env.P, env.R)

    obs = env.env_start()
    print(obs)
    ## test some observations and rewards
    for i in range(10):
        obs = env.env_step(np.random.choice(env.num_actions))
        print(obs[0], obs[1])


# to run this test, comment out all of the environments/__init__ file.
# the contents of that file are necessary for all the environments to be available in the outer folder's run_exp
if __name__ == '__main__':
    main()

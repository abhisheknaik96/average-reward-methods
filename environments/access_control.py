from environments.base_environment import BaseEnvironment
import numpy as np


class AccessControl(BaseEnvironment):
    """
    Implements the Access Control Queueing task as described in Chapter 10 by Sutton and Barto (2018).
    """

    def __init__(self):

        self.num_servers = 10
        self.num_priorities = 4
        self.priorities = None
        self.prob_server_free = None

        self.num_states = (self.num_servers + 1) * self.num_priorities
        self.num_actions = 2

        self.actions = None
        self.reward_obs_term = None

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.priorities = [2 ** i for i in range(self.num_priorities)]
        self.prob_server_free = env_info.get('prob_server_free', 0.06)  # probability a server becomes free at every timestep

        self.actions = [0, 1]

        self.rand_generator = np.random.RandomState(env_info.get('random_seed', 42))
        self.obs_type = env_info.get('obs_type', 'one-hot')
        assert self.obs_type in ["full_state", "one-hot"]
        self.reward_obs_term = [0.0, None, False]

        self.counts = np.zeros(((self.num_servers + 1), self.num_priorities))

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.num_free_servers = self.num_servers
        self.current_request_priority = self.rand_generator.choice(self.priorities)
        self.counts[self.num_free_servers][int(np.log2(self.current_request_priority))] += 1

        observation = self.get_obs_from_state(self.num_free_servers, self.current_request_priority, self.obs_type)
        self.reward_obs_term[1] = observation

        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if terminal.
        """

        num_busy_servers = self.num_servers - self.num_free_servers
        for i in range(num_busy_servers):
            if self.rand_generator.rand() < self.prob_server_free:
                self.num_free_servers = min(self.num_servers, self.num_free_servers+1)

        reward = 0
        if action==1:
            if self.num_free_servers>0:
                reward = self.current_request_priority
                self.num_free_servers -= 1

        new_priority = self.rand_generator.choice(self.priorities)
        observation = self.get_obs_from_state(self.num_free_servers, new_priority, self.obs_type)
        self.counts[self.num_free_servers][int(np.log2(new_priority))] += 1
        self.current_request_priority = new_priority

        self.reward_obs_term = [reward, observation, False]

        return self.reward_obs_term

    def get_obs_from_state(self, num_free_servers, priority, obs_type):

        priority_idx = np.log2(priority)
        assert priority_idx in np.arange(self.num_priorities)
        # assert priority_idx % 1 == 0
        obs = None
        if obs_type == "full_state":
            # obs = [num_free_servers, priority]
            obs = num_free_servers * self.num_priorities + int(priority_idx)
        elif obs_type == "one-hot":
            obs = np.zeros(self.num_states)
            idx = num_free_servers * self.num_priorities + int(priority_idx)
            obs[idx] = 1

        return obs


def main():

    env = AccessControl()
    env.env_init()
    ## test the transition and reward matrices
    # print(env.P, env.R)

    obs = env.env_start()
    print(obs)
    ## test some observations and rewards
    for i in range(10):
        action = np.random.choice(env.num_actions)
        obs = env.env_step(action)
        print(action)
        print(obs[0], obs[1])


# to run this test, comment out all of the environments/__init__ file.
# the contents of that file are necessary for all the environments to be available in the outer folder's run_exp
if __name__ == '__main__':
    main()
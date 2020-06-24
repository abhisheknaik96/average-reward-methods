from environments.base_environment import BaseEnvironment, OneHotEnv
import numpy as np

# ToDo: reorganize the ones other than RiverSwim based on OOP inheritance principles


class RiverSwim(OneHotEnv):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):
        super().__init__()

        self.num_states = 6
        self.num_actions = 2

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.1
                self.P[s, 1, s + 1] = 0.9
                self.R[s, 0, s] = 5. / 1000.
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.95
                self.R[s, 1, s] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.05
                self.P[s, 1, s + 1] = 0.9

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]


class RiverSwim_Harder(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.num_states = 6
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))

        self.start_state = None

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.4
                self.P[s, 1, s + 1] = 0.6
                self.R[s, 0] = 5. / 1000.
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.4
                self.P[s, 1, s] = 0.6
                self.R[s, 1] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s + 1] = 0.35

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.current_state

        return self.reward_obs_term[1]


    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        cond_p = self.P[self.current_state, action]
        next_state = self.rand_generator.choice(self.num_states, 1, p=cond_p).item()
        assert cond_p[next_state] > 0.

        reward = self.R[self.current_state, action]
        self.current_state = next_state

        self.reward_obs_term = [reward, self.current_state, False]

        return self.reward_obs_term


class RiverSwim_Small(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.num_states = 4
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))

        self.start_state = None

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.7
                self.P[s, 1, s + 1] = 0.3
                self.R[s, 0] = 5. / 1000.
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.2
                self.P[s, 1, s] = 0.8
                self.R[s, 1] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.2
                self.P[s, 1, s] = 0.5
                self.P[s, 1, s + 1] = 0.3

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.current_state

        return self.reward_obs_term[1]


    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        cond_p = self.P[self.current_state, action]
        next_state = self.rand_generator.choice(self.num_states, 1, p=cond_p).item()
        assert cond_p[next_state] > 0.

        reward = self.R[self.current_state, action]
        self.current_state = next_state

        self.reward_obs_term = [reward, self.current_state, False]

        return self.reward_obs_term


class RiverSwim_Loop(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.num_states = 4
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        self.start_state = None

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.1
                self.P[s, 1, s + 1] = 0.9
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.45
                self.P[s, 1, 0] = 0.5   # loop back to leftmost state
                self.R[s, 1, s] = 5
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.05
                self.P[s, 1, s + 1] = 0.9

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.current_state

        return self.reward_obs_term[1]


    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        cond_p = self.P[self.current_state, action]
        next_state = self.rand_generator.choice(self.num_states, 1, p=cond_p).item()
        assert cond_p[next_state] > 0.

        reward = self.R[self.current_state, action, next_state]
        self.current_state = next_state

        self.reward_obs_term = [reward, self.current_state, False]

        return self.reward_obs_term


class SimpleRiverSwim(RiverSwim):

    def env_init(self, agent_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s + 1] = 1
                self.R[s, 0] = 5. / 1000.
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s] = 1
                self.R[s, 1] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s + 1] = 1

        self.start_state = 0
        self.reward_obs_term = [0.0, None, False]
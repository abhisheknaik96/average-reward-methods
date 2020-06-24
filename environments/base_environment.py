#!/usr/bin/env python

"""Abstract environment base class for RL-Glue-py.
"""

from __future__ import print_function
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseEnvironment:
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.num_actions = None
        self.num_states = None

    @abstractmethod
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    def one_hot_from_tabular(self, state):
        """Returns a one-hot representation of a given state
            E.g., for 5 states, 2 returns [0,0,1,0,0]
        Args:
            state (int) : state index
        Returns:
            (np.array)  : one-hot representation of the state
        """
        one_hot_state = np.zeros(self.num_states)
        one_hot_state[state] = 1
        return one_hot_state


class OneHotEnv(BaseEnvironment):
    """
    A template environment that returns one-hot vectors as the state representation
    """

    def __init__(self):
        super().__init__()

        self.P = None
        self.R = None

        self.start_state = 0
        self.current_state = 0

        self.random_seed = None
        self.rand_generator = None

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal. Set up the transition and reward dynamics matrices.
        """

        # initialize P and R
        # self.P =
        # self.R =

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]

        raise NotImplementedError

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.one_hot_from_tabular(self.current_state)

        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        cond_p = self.P[self.current_state][action]
        next_state = self.rand_generator.choice(self.num_states, 1, p=cond_p).item()
        assert cond_p[next_state] > 0.

        reward = self.R[self.current_state, action, next_state]
        self.current_state = next_state

        self.reward_obs_term = [reward, self.one_hot_from_tabular(self.current_state), False]

        return self.reward_obs_term

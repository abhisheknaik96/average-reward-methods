#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.num_actions = None
        self.num_states = None

    @abstractmethod
    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    def get_representation(self, observation, action=-1):
        """returns the representation.
        Args:
            observation (ndarray)  : the observation to be processed
            action (int)            : if -1, assumes state values are being used and returns the representation as is
                                     otherwise returns a copy of the representation at the index specified by the action
                                     e.g., ([1,2], 1) returns [0,0,1,2,0,0] for self.num_actions=3
                                          ([1,2],-1) returns [1,2]
                                     (note: discrete actions assumed)
        Returns:
            rep : ndarray
                the representation vector
        """
        if action == -1:
            return observation
        else:
            rep = np.zeros(self.num_states * self.num_actions)
            rep[self.num_states * action: self.num_states * (action + 1)] = observation
            return rep

from agents.base_agent import BaseAgent
import numpy as np
from utils.helpers import get_weights_from_npy


class LFAPredictionAgent(BaseAgent):
    """
    A generic class that is re-used in the implementation of all sorts of control algorithms with LFA.
    Only agent_step needs to be implemented in the child classes.
    """

    def __init__(self, config):
        super().__init__()
        self.num_actions = config['num_actions']
        self.num_states = config['num_states']          # this could also be the size of the observation vector

        self.alpha_w = None
        self.alpha_r = None
        self.value_init = None
        self.avg_reward_init = None

        self.weights = None
        self.avg_reward = None
        self.avg_value = None

        # ToDo: use a better way to specify the target and behavioural policies
        self.pi = None
        self.b = None

        self.rand_generator = None

        self.past_rho = None
        self.actions = None
        self.past_action = None
        self.past_state = None
        self.timestep = None
        self.error = None

    def choose_action(self, observation):
        """returns an action based on a pre-determined policy.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned policy
        # ToDo: at some point, assign choose_action a separate method based on agent_info["policy"],
        # just like in control_agents
        """

        action = self.rand_generator.choice(self.actions, p=self.b)
        # action = 1

        return action

    def set_weights(self, given_weight_vector):
        """sets the agent's weights to the given weight vector.
        Usually used for evaluating the learned policy"""
        self.weights = given_weight_vector

    def get_value(self, representation):
        """returns the action value linear in the representation and the weights
        Args:
            representation : ndarray
                the 'x' part of (w^T x)
        Returns:
            w^T x : float
        """
        return np.dot(representation, self.weights)

    def validate_behavioural_policy(self):
        okay = 1
        for i in range(len(self.pi)):
            if self.pi[i] > 0 and self.b[i] == 0:
                okay = 0
                break
        assert okay == 1, 'Behavioural policy does not have coverage'

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

        # assert "num_actions" in agent_info
        # self.num_actions = agent_info.get("num_actions", 4)
        # assert "num_states" in agent_info
        # self.num_states = agent_info["num_states"]
        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 22))

        self.alpha_w = agent_info.get("alpha_w", 0.1)
        self.eta = agent_info.get("eta", 1)
        # self.alpha_r = agent_info.get("alpha_r", self.alpha_w)
        self.alpha_r = self.eta * self.alpha_w
        self.value_init = agent_info.get("value_init", 0)
        self.avg_reward_init = agent_info.get("avg_reward_init", 0)

        # self.weights = self.rand_generator.normal(0, 0.1, self.num_states) + self.value_init
        self.weights = np.zeros(self.num_states) + self.value_init
        self.avg_reward = 0.0 + self.avg_reward_init
        ### if the weights are supplied, they should be an ndarray in an npy file
        ### as a dictionary element with key 'weights'
        if 'weights_file' in agent_info:
            weights = get_weights_from_npy(agent_info['weights_file'])
            assert weights.size == self.num_states * self.num_actions
            self.weights = weights
            # if avg_reward != None
            #     assert np.ndim(avg_reward) == 0
            # self.avg_reward = avg_reward

        assert 'pi' in agent_info
        self.pi = agent_info['pi']
        self.b = agent_info.get('b', self.pi)
        self.validate_behavioural_policy()

        self.avg_value = 0.0
        self.actions = list(range(self.num_actions))
        self.past_action = None
        self.past_state = None
        self.timestep = 0  # for debugging

    def agent_start(self, observation):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            (integer) the first action the agent takes.
        """

        self.past_action = self.choose_action(observation)
        self.past_state = self.get_representation(observation, -1)
        self.past_rho = self.pi[self.past_action] / self.b[self.past_action] if self.past_state[0]==1 else 1.0
        self.timestep += 1

        return self.past_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (integer) The action the agent takes given this observation.
        """
        raise NotImplementedError

    def agent_end(self, reward):
        """Run when the agent terminates.
        A direct-RL update with the final transition. Not applicable for continuing tasks
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass


class DifferentialTDAgent(LFAPredictionAgent):
    """
    Implements the newly-proposed Differential TD-learning algorithm.
    """

    def __init__(self, config):
        super().__init__(config)

        self.weights_f = None
        self.average_value = None
        self.alpha_w_f = None
        self.alpha_r_f = None

    def get_value_f(self, representation):
        """returns the higher-order action value linear in the representation and the weights
        Args:
            representation : ndarray
                the 'x' part of (w^T x)
        Returns:
            w^T x : float
        """
        return np.dot(representation, self.weights_f)

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        # self.weights_f = self.rand_generator.normal(0, 0.1, self.num_states)
        self.weights_f = np.zeros(self.num_states)
        self.avg_value = 0.0
        self.alpha_w_f = agent_info.get("alpha_w_f", 0.1)
        self.alpha_r_f = agent_info.get("alpha_r_f", self.alpha_w_f)

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (integer) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value in the agent parameters agent_info
        """
        action = self.choose_action(observation)
        state = self.get_representation(observation, -1)
        delta = reward - self.avg_reward + self.get_value(state) - self.get_value(self.past_state)
        self.weights += self.alpha_w * self.past_rho * delta * self.past_state
        self.avg_reward += self.alpha_r * self.past_rho * delta
        self.error = delta
        delta_f = self.get_value(self.past_state) - self.avg_value + \
                  self.get_value_f(state) - self.get_value_f(self.past_state)
        self.weights_f += self.alpha_w_f * self.past_rho * delta_f * self.past_state
        self.avg_value += self.alpha_r_f * self.past_rho * delta_f

        self.past_state = state
        self.past_action = action
        # ToDo: very hacky and specific to state 0's actions in TwoLoop; make more general
        self.past_rho = self.pi[self.past_action] / self.b[self.past_action] if self.past_state[0]==1 else 1.0

        return self.past_action


class AvgCostTDAgent(LFAPredictionAgent):
    """
    Implements Tsitsklis and Van Roy (1999)'s AvgCostTD algorithm
    """

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (integer) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value by default unless otherwise specified in agent_info
        """
        action = self.choose_action(observation)
        state = self.get_representation(observation, -1)
        delta = reward - self.avg_reward + self.get_value(state) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state
        self.error = (reward - self.avg_reward)
        self.avg_reward += self.alpha_r * (reward - self.avg_reward)

        self.past_state = state
        self.past_action = action

        return self.past_action


def test_DiffTDoff():

    agent = DifferentialTDAgent({'num_states': 3, 'num_actions': 2})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5, 'pi': [0.5, 0.5], 'b': [0.9, 0.1]})
    agent.weights = np.array([1, 2, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.past_state, action, agent.past_rho)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action, agent.past_rho)


def test_DiffTD():

    agent = DifferentialTDAgent({'num_states': 3, 'num_actions': 2})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5, 'pi': [0.5, 0.5]})
    agent.weights = np.array([1, 2, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.past_state, action, agent.past_rho)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action, agent.past_rho)


if __name__ == '__main__':
    test_DiffTD()
    # test_DiffTDoff()


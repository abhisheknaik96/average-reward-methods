from agents.base_agent import BaseAgent
import numpy as np
from utils.helpers import get_weights_from_npy
from utils.tilecoding import TileCoder

class LFAControlAgent(BaseAgent):
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
        self.epsilon = None
        self.choose_action = None                       # the policy (e-greedy/greedy/random)
        self.max_action = None                          # the greedy action for Q-learning-esque update

        self.tilecoder = None
        self.weights = None
        self.avg_reward = None
        self.avg_value = None

        self.rand_generator = None

        self.actions = None
        self.past_action = None
        self.past_state = None
        self.timestep = None

    def choose_action_egreedy(self, observation):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
            action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())

        return action

    def choose_action_greedy(self, observation):
        """returns an action using a greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned greedy policy
        """
        q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
        action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())

        return action

    def choose_action_random(self, observation):
        """returns a random action indifferent to the current action-value function.
        Args:
        Returns:
            (Integer) The action taken
        """
        action = self.rand_generator.choice(self.actions)
        return action

    def pick_policy(self, policy_type):
        """returns the method that'll pick actions based on the argument"""
        if policy_type == 'random':
            return self.choose_action_random
        elif policy_type == 'greedy':
            return self.choose_action_greedy
        elif policy_type == 'egreedy':
            return self.choose_action_egreedy

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

    def max_action_value(self, observation):
        """returns the action corresponding to the maximum action value for the given observation"""
        q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
        self.max_action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())

        return q_s[self.max_action]

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

        # assert "num_actions" in agent_info
        # self.num_actions = agent_info.get("num_actions", 4)
        # assert "num_states" in agent_info
        if "num_states" in agent_info:
            self.num_states = agent_info["num_states"]  # if this needs to be over-written
        self.bias = agent_info.get("bias", False)
        if self.bias:
            self.num_states += 1
        self.tilecoder = agent_info.get("tilecoder", False)
        if self.tilecoder:
            num_tilings = agent_info.get("num_tilings", 8)
            assert "dims" in agent_info
            dims = agent_info["dims"]
            assert "limits" in agent_info
            limits = agent_info["limits"]

            self.tilecoder = TileCoder(dims=dims, limits=limits, tilings=num_tilings,
                                       style='vector')
            self.num_states = self.tilecoder.n_tiles

        self.alpha_w = agent_info.get("alpha_w", 0.1)
        self.eta = agent_info.get("eta", 1)
        # self.alpha_r = agent_info.get("alpha_r", self.alpha_w)
        self.alpha_r = self.eta * self.alpha_w
        self.value_init = agent_info.get("value_init", 0)
        self.avg_reward_init = agent_info.get("avg_reward_init", 0)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.choose_action = self.pick_policy(agent_info.get("policy_type", "egreedy"))

        self.weights = np.zeros(self.num_states * self.num_actions) + self.value_init
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
        # self.bias = np.zeros(self.num_actions) + self.value_init
        # self.delta = 0

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 22))

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
        if self.tilecoder:
            observation = self.tilecoder.__getitem__(observation)
        self.past_action = self.choose_action(observation)
        self.past_state = self.get_representation(observation, self.past_action)
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


class DifferentialQlearningAgent_v1(LFAControlAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, config):
        super().__init__(config)

        self.weights_f = None
        self.average_value = None
        self.alpha_w_f = None
        self.alpha_r_f = None
        self.max_action = None

    def get_value_f(self, representation):
        """returns the higher-order action value linear in the representation and the weights
        Args:
            representation : ndarray
                the 'x' part of (w^T x)
        Returns:
            w^T x : float
        """
        return np.dot(representation, self.weights_f)

    def max_action_value_f(self, observation):
        """
        returns the higher-order action value corresponding to the
        maximum lower-order action value for the given observation.
        Note: this is not max_a q_f(s,a)
        """
        # q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
        # max_action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())
        #
        # q_f_s = np.array([self.get_value_f(self.get_representation(observation, a)) for a in self.actions])
        # return q_f_s[max_action]

        q_f_sa = self.get_value_f(self.get_representation(observation, self.max_action))
        return q_f_sa

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        self.weights_f = np.zeros((self.num_states * self.num_actions))
        self.avg_value = 0.0
        self.alpha_w_f = agent_info.get("alpha_w_f", 0.1)
        self.eta_f = agent_info.get("eta_f", 1)
        # self.alpha_r_f = agent_info.get("alpha_r_f", self.alpha_w_f)
        self.alpha_r_f = self.eta_f * self.alpha_w_f

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
        if self.tilecoder:
            observation = self.tilecoder.__getitem__(observation)

        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state
        # self.avg_reward += self.beta * (reward - self.avg_reward)
        self.avg_reward += self.alpha_r * delta
        delta_f = self.get_value(self.past_state) - self.avg_value + \
                  self.max_action_value_f(observation) - self.get_value_f(self.past_state)
        self.weights_f += self.alpha_w_f * delta_f * self.past_state
        self.avg_value += self.alpha_r_f * delta_f

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.past_state = state
        self.past_action = action
        # self.step_size *= 0.9995
        # self.beta *= 0.9995

        return self.past_action

    def planning_update(self, obs, action, reward, obs_next):
        feature_vec = self.get_representation(obs, action)
        delta = reward - self.avg_reward + self.max_action_value(obs_next) - self.get_value(feature_vec)
        self.weights += self.alpha_w * delta * feature_vec
        self.avg_reward += self.alpha_r * delta


class DifferentialQlearningAgent_v2(DifferentialQlearningAgent_v1):
    """
    Extends the newly-proposed Differential Q-learning algorithm
    such that centering affects the learning process
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
        """
        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * (delta - self.avg_value) * self.past_state
        self.avg_reward += self.alpha_r * delta
        delta_f = self.get_value(self.past_state) - self.avg_value + \
                  self.max_action_value_f(observation) - self.get_value_f(self.past_state)
        self.weights_f += self.alpha_w_f * delta_f * self.past_state
        self.avg_value += self.alpha_r_f * delta_f

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.past_state = state
        self.past_action = action

        return self.past_action


class RVIQlearningAgent(LFAControlAgent):
    """
    Implements a version of the RVI Q-learning algorithm (Abounadi et al., 2001)
    with f as the value of a fixed state-action pair.
    self.avg_reward is used as a non-learned parameter in place of f in the algorithm.
    """

    def __init__(self, config):
        super().__init__(config)
        self.f_type = None
        self.reference_s = None     # raw state index
        self.reference_a = None     # raw action index
        self.reference_sa_rep = None    # representation corresponding to this s-a pair
        self.reference_s_rep = None     # representation corresponding to just this reference_s
        self.max_val = None             # for tracking the maximum action value when f_type=max_all_sa
        self.max_val_rep = None         # for tracking the representation corresponding to the above
        self.last_updated_rep = None    # required for max_all_sa computation
        self.f_sample_mean = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        self.f_type = agent_info.get('f_type', 'reference_sa')
        self.reference_s = agent_info.get('reference_state', 0)
        self.reference_a = agent_info.get('reference_action', 0)
        self.max_val = agent_info.get('max_val_init', -np.inf)

    def setup_f(self, observation):
        # this means state representation is tabular/one-hot
        if self.f_type == 'reference_sa':
            self.reference_s_rep = np.zeros(observation.shape)
            self.reference_s_rep[self.reference_s] = 1
            self.reference_sa_rep = self.get_representation(self.reference_s_rep, self.reference_a)
        # also implies state representation is tabular/one-hot
        elif self.f_type == 'max_a_reference_s':
            self.reference_s_rep = np.zeros(observation.shape)
            self.reference_s_rep[self.reference_s] = 1
        # either tabular representation or arbitrary features
        elif self.f_type == 'first_s':
            self.reference_s_rep = observation
            self.reference_sa_rep = self.get_representation(self.reference_s_rep, self.reference_a)
        # either tabular representation or arbitrary features
        # setup-wise, same as first_s with an arbitrary s and a.
        elif self.f_type == 'max_all_sa':
            self.reference_s_rep = observation
            self.last_updated_rep = self.get_representation(self.reference_s_rep, self.reference_a)
        # correctly computes the mean for one-hot features;
        # is a gross approximation in case of LFA with binary features
        elif self.f_type == 'mean':
            num_weights = self.weights.size
            self.reference_sa_rep = np.ones(num_weights) * 1./num_weights
        # # better approximation of mean of all Q(s,a)
        # elif self.f_type == 'sample_mean':
        #     self.f_sample_mean = 0
        elif self.f_type == 'all_ones':
            self.reference_sa_rep = np.ones(self.weights.shape)
        elif self.f_type == 'random_bernoulli':
            self.reference_sa_rep = self.rand_generator.binomial(n=1, p=0.5, size=self.weights.shape)

    def get_f_value(self):
        f = None    # the value that shall be returned
        if self.f_type == 'reference_sa':
            f = self.get_value(self.reference_sa_rep)
        elif self.f_type == 'max_a_reference_s':
            f = self.max_action_value(self.reference_s_rep)
        elif self.f_type == 'first_s':
            f = self.get_value(self.reference_sa_rep)
        elif self.f_type == 'max_all_sa':
            ### self.max_val = max(self.max_val, self.get_value(self.past_state))
            past_val = self.get_value(self.last_updated_rep)
            # if the newly-computed value is larger, update
            if self.max_val < past_val:
                self.max_val = past_val
                self.max_val_rep = self.last_updated_rep
            # if it isn't, but it is corresponding to the s-a pair that had yielded the max previously,
            # update the max as the value of that s-a pair might have decreased
            elif (self.max_val_rep == self.last_updated_rep).all():
                self.max_val = past_val
            f = self.max_val
        elif self.f_type == 'mean':
            f = self.get_value(self.reference_sa_rep)
        elif self.f_type == 'all_ones':
            f = self.get_value(self.reference_sa_rep)
        elif self.f_type == 'random_bernoulli':
            f = self.get_value(self.reference_sa_rep)
        return f

    def agent_start(self, observation):
        if self.tilecoder:
            observation = self.tilecoder.__getitem__(observation)
        self.past_action = self.choose_action(observation)
        self.past_state = self.get_representation(observation, self.past_action)
        self.setup_f(observation)
        return self.past_action

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
        """
        if self.tilecoder:
            observation = self.tilecoder.__getitem__(observation)

        self.avg_reward = self.get_f_value()
        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.last_updated_rep = self.past_state
        self.past_state = state
        self.past_action = action

        return self.past_action


class RlearningAgent(LFAControlAgent):
    """
    Implements the R-learning algorithm by Schwartz (1993).
    """

    def __init__(self, config):
        super().__init__(config)
        # self.past_max_actions holds all the greedy actions for the previous observation
        self.past_max_actions = None

    def agent_start(self, observation):
        action = super().agent_start(observation)
        self.past_max_actions = self.get_max_actions(observation)
        return action

    def get_max_actions(self, observation):
        q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
        max_actions = np.argwhere(q_s == np.amax(q_s))
        return max_actions

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
        """
        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state
        if self.past_action in self.past_max_actions:
            self.avg_reward += self.alpha_r * delta

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.past_max_actions = self.get_max_actions(observation)
        self.past_state = state
        self.past_action = action

        return self.past_action


class Algo3Agent(LFAControlAgent):
    """
    Implements Algorithm 3 by Singh (1994).
    The idea of setting the value of a reference s-a pair to zero
        is restricted to the tabular case. It has been made possible here
        based on the assumption that the representation is one-hot,
        and hence equivalent to tabular.
    """

    def __init__(self, config):
        super().__init__(config)
        self.reference_s = None     # raw state index
        self.reference_a = None     # raw action index
        self.reference_sa_idx = None    # index in the weight vector corresponding to the ref s-a

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.reference_s = agent_info.get('reference_state', 0)
        self.reference_a = agent_info.get('reference_action', 0)

    def agent_start(self, observation):
        action = super().agent_start(observation)
        ref_state = np.zeros(observation.shape)
        ref_state[self.reference_s] = 1
        reference_sa = self.get_representation(ref_state, self.reference_a)
        self.reference_sa_idx = np.where(reference_sa == 1)[0][0]

        return action

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
        """
        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state
        self.avg_reward += self.alpha_r * delta

        self.weights[self.reference_sa_idx] = 0

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.past_obs = observation
        self.past_state = state
        self.past_action = action

        return self.past_action


# class ActorCriticAgent(ReinforceAgent):
#     """
#     Episodic/Continuing Actor-Critic for control
#     """
#
#     def __init__(self, agent_info={}):
#         super().__init__(agent_info)
#
#         self.w = np.zeros(self.num_features)
#         self.alpha_v = agent_info.get('alpha_v', 2e-2)
#
#         self.past_obs = None
#         self.past_action = None
#         self.gamma_decay = None
#
#     def start(self, observation):
#         """called for the first action given an observation"""
#         action = self.choose_action(observation)
#         self.past_obs = observation
#         self.past_action = action
#         self.gamma_decay = 1
#         return action
#
#     def step(self, reward, observation):
#         """returns an action given an observation, updates the actor and critic parameters"""
#
#         tmp = np.array([self.get_sa_representation(self.past_obs, a) for a in range(self.num_actions)])
#         pi = self.get_pi(self.past_obs)
#         grad_ln = self.get_sa_representation(self.past_obs, self.past_action) - np.dot(pi, tmp)
#         v = np.dot(self.w, self.get_s_representation(self.past_obs))
#         v_next = np.dot(self.w, self.get_s_representation(observation))
#         delta = reward + self.gamma*v_next - v
#         grad_v = self.get_s_representation(self.past_obs)
#
#         self.theta += self.alpha * self.gamma_decay * delta * grad_ln
#         self.w += self.alpha_v * self.gamma_decay * delta * grad_v
#
#         self.gamma_decay *= self.gamma
#         action = self.choose_action(observation)
#         return action
#
#     def end(self, reward):
#         """updates the actor and critic parameters for the final time in an episode (not called for continuing tasks)"""
#
#         tmp = np.array([self.get_sa_representation(self.past_obs, a) for a in range(self.num_actions)])
#         pi = self.get_pi(self.past_obs)
#         grad_ln = self.get_sa_representation(self.past_obs, self.past_action) - np.dot(pi, tmp)
#         v = np.dot(self.w, self.get_s_representation(self.past_obs))
#         delta = reward - v
#         grad_v = self.get_s_representation(self.past_obs)
#
#         self.theta += self.alpha * self.gamma_decay * delta * grad_ln
#         self.w += self.alpha_v * self.gamma_decay * delta * grad_v


# Tests

def test_DiffQ():

    agent = DifferentialQlearningAgent_v1({'num_states': 3, 'num_actions': 3, 'alpha_w_f': 0.0})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
    agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1, 1, 0, 0]) * 1.0
    observation = np.array([1, -1, 0.5])
    action = agent.agent_start(observation)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action)


def test_RVIQ():

    agent = RVIQlearningAgent({'num_states': 3, 'num_actions': 3})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5, 'f_type': 'max_all_sa'})
    agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    # print(agent.reference_sa)
    print(agent.max_val)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.max_val)
        print(agent.past_state, agent.past_action)


def test_R():

    agent = RlearningAgent({'num_states': 3, 'num_actions': 3})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
    agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action)
        print(agent.weights)


def test_Algo3():

    agent = Algo3Agent({'num_states': 3, 'num_actions': 3})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5,
                      'reference_state': 0, 'reference_action': 2})
    agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.reference_sa_idx)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action)


def test_DiffQ_planning_update():

    agent = DifferentialQlearningAgent_v1({'num_states': 3, 'num_actions': 2, 'alpha_w_f': 0.0})
    agent.agent_init({'random_seed': 32})
    agent.weights = np.zeros(6)

    for i in range(5):
        obs = np.random.binomial(n=1, p=0.5, size=3)
        action = np.random.choice(2)
        reward = 1
        obs_next = np.random.binomial(n=1, p=0.5, size=3)
        print(obs, action, reward, obs_next)
        agent.planning_update(obs, action, reward, obs_next)
        print(agent.weights)
        print(agent.avg_reward)
        print()


if __name__ == '__main__':
    # test_DiffQ()
    # test_RVIQ()
    # test_R()
    # test_Algo3()
    test_DiffQ_planning_update()

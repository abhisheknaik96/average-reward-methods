"""
PuckWorld Environment
Original Author: Qiang Ye (https://github.com/qqiang00/ReinforcemengLearningPractice/blob/master/reinforce/puckworld.py)
Adapted to RL-glue by Abhishek Naik in Jan 2020
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time

RAD2DEG = 57.29577951308232


class PuckWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.num_states = 6     # actually, num_features
        self.num_actions = 4    # not considering the no-op action

    def env_init(self, env_info={}):
        self.width = 256
        self.height = 256
        self.l_unit = 1.0
        self.v_unit = 1.0
        self.max_speed = 0.025      # max agent velocity along an axis

        self.accel = 0.002          # agent acceleration
        self.rad = env_info.get('goal_update_time', 0.1)            # radius of agent
        self.target_rad = env_info.get('goal_update_time', 0.02)    # radius of target
        self.goal_dis = self.rad    # minimum distance to target to get a positive reward
        self.t = 0                  # puck world clock
        self.update_time = env_info.get('goal_update_time', 100)      # time for target randomize its position
        # Bounds on the observation space
        self.low = np.array([0,                 # agent position x
                             0,                 # agent position y
                             -self.max_speed,   # agent velocity x
                             -self.max_speed,   # agent velocity y
                             0,                 # target position x
                             0,                 # target position y
                             ])
        self.high = np.array([self.l_unit,
                              self.l_unit,
                              self.max_speed,
                              self.max_speed,
                              self.l_unit,
                              self.l_unit,
                              ])
        self.action = None  # for rendering
        self.viewer = None
        self.action_space = spaces.Discrete(self.num_actions)  # 0,1,2,3,4 represent left, right, up, down, no-op
        self.observation_space = spaces.Box(self.low, self.high)
        self.rand_generator = np.random.RandomState(env_info.get('random_seed', 42))
        self.render = env_info.get('render', False)
        self.reward_obs_term = [0.0, None, False]

        self.game_state = np.zeros(self.low.shape)

    def env_start(self):
        self.game_state = np.array([self.random_pos(),
                               self.random_pos(),
                               0,
                               0,
                               self.random_pos(),
                               self.random_pos()
                               ])
        observation = self.process_raw_obs(self.game_state)
        self.reward_obs_term[1] = observation
        return self.reward_obs_term[1]

    def env_step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action                # action for rendering
        ppx, ppy, pvx, pvy, tx, ty = np.ndarray.tolist(self.game_state)
        ppx, ppy = ppx + pvx, ppy + pvy     # update agent position
        pvx, pvy = pvx * 0.95, pvy * 0.95   # natural velocity loss

        if action == 0: pvx -= self.accel   # left
        if action == 1: pvx += self.accel   # right
        if action == 2: pvy += self.accel   # up
        if action == 3: pvy -= self.accel   # down
        if action == 4: pass  # no move

        if ppx < self.rad:          # encounter left bound
            pvx *= -0.5
            ppx = self.rad
        if ppx > 1 - self.rad:      # right bound
            pvx *= -0.5
            ppx = 1 - self.rad
        if ppy < self.rad:          # bottom bound
            pvy *= -0.5
            ppy = self.rad
        if ppy > 1 - self.rad:      # top bound
            pvy *= -0.5
            ppy = 1 - self.rad

        self.t += 1
        if self.t % self.update_time == 0:  # update target position
            tx = self.random_pos()          #   randomly
            ty = self.random_pos()

        dx, dy = ppx - tx, ppy - ty     # distance from
        dis = self.compute_dis(dx, dy)  #   agent to target

        reward = self.goal_dis - dis    # distance from agent to target

        self.game_state = np.array((ppx, ppy, pvx, pvy, tx, ty))
        observation = self.process_raw_obs(self.game_state)
        self.reward_obs_term = [reward, observation, False]

        if self.render:
            self.render_screen()

        return self.reward_obs_term

    def random_pos(self):
        return self.rand_generator.uniform(low=0, high=self.l_unit)

    def compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    def process_raw_obs(self, obs_raw):
        """
        converts the dictionary into a usable list with scaled features
        Rough ranges:
            - agent_x     : [0, self.l_unit]
            - agent_y     : [0, self.l_unit]
            - agent_vel_x : [-self.max_speed, self.max_speed]
            - agent_vel_y : [-self.max_speed, self.max_speed]
            - target_x    : [0, self.l_unit]
            - target_y    : [0, self.l_unit]

        Scaled ranges:
            - agent_x     : [0, 1]
            - agent_y     : [0, 1]
            - agent_vel_x : [-1, 1]
            - agent_vel_y : [-1, 1]
            - target_x    : [0, 1]
            - target_y    : [0, 1]
        """

        obs = np.zeros(obs_raw.shape)
        obs[0] = obs_raw[0] / self.l_unit
        obs[1] = obs_raw[1] / self.l_unit
        obs[2] = obs_raw[2] / self.max_speed
        obs[3] = obs_raw[3] / self.max_speed
        obs[4] = obs_raw[4] / self.l_unit
        obs[5] = obs_raw[5] / self.l_unit

        return obs

    def render_screen(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale = self.width / self.l_unit
        rad = self.rad * scale
        t_rad = self.target_rad * scale  # target radius

        # If the screen object has not been set, initialize the elements of the entire screen.
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            target = rendering.make_circle(t_rad, 30, True)
            target.set_color(0.1, 0.9, 0.1)
            self.viewer.add_geom(target)
            target_circle = rendering.make_circle(t_rad, 30, False)
            target_circle.set_color(0, 0, 1)
            self.viewer.add_geom(target_circle)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target_circle.add_attr(self.target_trans)

            self.agent = rendering.make_circle(rad, 30, True)
            self.agent.set_color(0, 1, 0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            agent_circle = rendering.make_circle(rad, 30, False)
            agent_circle.set_color(0, 0, 0)
            agent_circle.add_attr(self.agent_trans)
            self.viewer.add_geom(agent_circle)

            self.line_trans = rendering.Transform()
            self.arrow = rendering.FilledPolygon([
                (0.7 * rad, 0.15 * rad),
                (rad, 0),
                (0.7 * rad, -0.15 * rad)
            ])
            self.arrow.set_color(0, 0, 0)
            self.arrow.add_attr(self.line_trans)
            self.viewer.add_geom(self.arrow)

        ppx, ppy, _, _, tx, ty = np.ndarray.tolist(self.reward_obs_term[1])
        self.target_trans.set_translation(tx * scale, ty * scale)
        self.agent_trans.set_translation(ppx * scale, ppy * scale)
        # Coloring agent by distance
        vv, ms = self.reward_obs_term[0] + 0.3, 1
        r, g, b, = 0, 1, 0
        if vv >= 0:
            r, g, b = 1 - ms * vv, 1, 1 - ms * vv
        else:
            r, g, b = 1, 1 + ms * vv, 1 + ms * vv
        self.agent.set_color(r, g, b)

        a = self.action
        if a in [0, 1, 2, 3]:
            # draw arrows based on action
            degree = 0
            if a == 0:
                degree = 180
            elif a == 1:
                degree = 0
            elif a == 2:
                degree = 90
            else:
                degree = 270
            self.line_trans.set_translation(ppx * scale, ppy * scale)
            self.line_trans.set_rotation(degree / RAD2DEG)
            # self.line.set_color(0,0,0)
            self.arrow.set_color(0, 0, 0)
        else:
            # self.line.set_color(r,g,b)
            self.arrow.set_color(r, g, b)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == "__main__":

    render = True
    game = PuckWorld()
    game.env_init({'random_seed': 42, 'render': render})
    obs = game.env_start()
    print(obs)

    for i in range(100):
        action = np.random.choice(4)
        print(action)
        reward_obs_term = game.env_step(action)
        print(reward_obs_term[0], reward_obs_term[1])
        if render:
            game.render_screen()
        time.sleep(0.5)
    game.close()
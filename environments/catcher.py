### Modified from https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/catcher.py

import sys
import pygame
import numpy as np
# from .utils import percent_round_int
import time

from ple.games import base
from pygame.constants import K_a, K_d


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


class Paddle(pygame.sprite.Sprite):

    def __init__(self, speed, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.speed = speed
        self.width = width

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.vel = 0.0

        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((width, height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, width, height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (
            SCREEN_WIDTH / 2 - self.width / 2,
            SCREEN_HEIGHT - height - 3)

    def update(self, dx, dt):
        self.vel += dx
        self.vel *= 0.9

        x, y = self.rect.center
        n_x = x + self.vel

        if n_x <= 0:
            self.vel = 0.0
            n_x = 0

        if n_x + self.width >= self.SCREEN_WIDTH:
            self.vel = 0.0
            n_x = self.SCREEN_WIDTH - self.width

        self.rect.center = (n_x, y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Fruit(pygame.sprite.Sprite):

    def __init__(self, speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng):
        self.speed = speed
        self.size = size

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.rng = rng

        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((size, size))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (0, 120, 120),
            (0, 0, size, size),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (-30, -30)

    def update(self, dt):
        x, y = self.rect.center
        dt = 30
        n_y = y + self.speed * dt

        self.rect.center = (x, n_y)

    def reset(self):
        x = self.rng.choice(
            range(
                self.size *
                2,
                self.SCREEN_WIDTH -
                self.size *
                2,
                self.size))
        y = self.rng.choice(
            range(
                self.size,
                int(self.SCREEN_HEIGHT / 2),
                self.size))

        self.rect.center = (x, -1 * y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


# CatcherBase combines Paddle and Fruit to make the minimal working env.
# But the following Catcher class provides an RL-Glue compliant implementation.
class CatcherBase(base.PyGameWrapper):
    """
    From https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/catcher.py

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_lives : int (default: 3)
        The number lives the agent has.

    """

    def __init__(self, width=64, height=64, init_lives=3, render=False):

        actions = {
            "left": K_a,
            "right": K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.fruit = None
        self.fruit_size = percent_round_int(height, 0.06)
        self.fruit_fall_speed = 0.00095 * height

        self.player = None
        self.player_speed = 0.021 * width
        self.paddle_width = percent_round_int(width, 0.2)
        self.paddle_height = percent_round_int(height, 0.04)

        self.dx = 0.0
        self.init_lives = init_lives
        self.render = render

    def _handle_player_events(self):
        self.dx = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions['left']:
                    self.dx -= self.player_speed

                if key == self.actions['right']:
                    self.dx += self.player_speed

    def get_dx_from_action(self, action):
        self.dx = 0.0
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if action == 0:  # left
            self.dx -= self.player_speed
        if action == 1:  # right
            self.dx += self.player_speed

    def init(self):
        self.score = 0
        self.player = Paddle(self.player_speed, self.paddle_width,
                             self.paddle_height, self.width, self.height)

        self.fruit = Fruit(self.fruit_fall_speed, self.fruit_size,
                           self.width, self.height, self.rng)

        self.fruit.reset()

    def get_game_state(self):
        """
        Gets a non-visual state representation of the game.
        Returns
        -------

        dict
            * player x position.
            * players velocity.
            * fruits x position.
            * fruits y position.

            See code for structure.

        """
        state = {
            "player_x": self.player.rect.center[0],
            "player_vel": self.player.vel,
            "fruit_x": self.fruit.rect.center[0],
            "fruit_y": self.fruit.rect.center[1]
        }

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == 0

    def step_old(self, dt):
        if self.render:
            self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        if self.fruit.rect.center[1] >= self.height:
            self.score += self.rewards["negative"]
            self.lives -= 1
            self.fruit.reset()

        if pygame.sprite.collide_rect(self.player, self.fruit):
            self.score += self.rewards["positive"]
            self.fruit.reset()

        self.player.update(self.dx, dt)
        self.fruit.update(dt)

        if self.lives == 0:
            self.score += self.rewards["loss"]

        if self.render:
            self.player.draw(self.screen)
            self.fruit.draw(self.screen)

    def step(self, action):
        if self.render:
            self.screen.fill((0, 0, 0))

        self.get_dx_from_action(action)
        # dt = self.clock.tick_busy_loop(30)
        # dt = self.clock.tick_busy_loop(100)
        # dt = self.clock.tick(10000)
        dt = 30

        self.score += self.rewards["tick"]

        if self.fruit.rect.center[1] >= self.height:
            self.score += self.rewards["negative"]
            self.fruit.reset()

        if pygame.sprite.collide_rect(self.player, self.fruit):
            self.score += self.rewards["positive"]
            self.fruit.reset()

        self.player.update(self.dx, dt)
        self.fruit.update(dt)

        if self.render:
            self.player.draw(self.screen)
            self.fruit.draw(self.screen)


class Catcher(CatcherBase):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.
    """

    def __init__(self, width=256, height=256):

        actions = {
            "left": K_a,
            "right": K_d
        }

        self.num_states = 4  # actually, num_features
        self.num_actions = 2

        self.height = height
        self.width = width

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.fruit_size = None
        self.fruit_fall_speed = None

        self.player_speed = None
        self.paddle_width = None
        self.paddle_height = None

        self.dx = None
        self.render = None
        self.random_seed = None
        self.rng = None

        self.reward_obs_term = None
        self.score = None
        self.prev_score = None
        self.player = None
        self.fruit = None

    def env_init(self, env_info={}):

        self.fruit_size = percent_round_int(self.height, 0.06)
        self.fruit_fall_speed = 0.00095 * self.height

        self.player_speed = 0.021 * self.width
        self.paddle_width = percent_round_int(self.width, 0.2)
        self.paddle_height = percent_round_int(self.height, 0.04)
        self.random_seed = env_info.get('random_seed', 42)
        self.rng = np.random.RandomState(self.random_seed)
        self.new_rewards = env_info.get('rewards', {"negative": -40.0, "positive": 40.0})
        self.adjustRewards(self.new_rewards)

        self.dx = 0.0
        self.render = env_info.get('render', False)

        if self.render:
            self.screen = pygame.display.set_mode(self.getScreenDims(), 0, 32)
        self.clock = pygame.time.Clock()

        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        self.init()
        self.prev_score = 0.0

        observation_raw = self.get_game_state()
        observation = self.process_raw_obs(observation_raw)
        self.reward_obs_term[1] = observation

        if self.render:
            pygame.display.update()

        return self.reward_obs_term[1]

    def env_step(self, action):

        self.step(action)
        reward = self.score - self.prev_score
        self.prev_score = self.score

        observation_raw = self.get_game_state()
        observation = self.process_raw_obs(observation_raw)
        self.reward_obs_term = [reward, observation, False]

        if self.render:
            pygame.display.update()

        return self.reward_obs_term

    def process_raw_obs(self, obs_raw):
        """
        converts the dictionary into a usable list with scaled features
        Rough ranges:
            - player_x  : [0, self.width]
            - player_vel: [-0.2*self.width, 0.2*self.width]
            - fruit_x   : [0, self.width]
            - fruit_y   : [-self.height/2, self.height]
        The upper bound for player_vel is computed assuming no exponential loss factor of 0.9

        Scaled ranges:
            - player_x  : [0, 1]
            - player_vel: [-1, 1]
            - fruit_x   : [0, 1]
            - fruit_y   : [-0.5, 1]
        """

        obs = np.zeros(self.num_states)
        obs[0] = obs_raw['player_x'] / self.width
        obs[1] = obs_raw['player_vel'] / (0.2 * self.width)
        obs[2] = obs_raw['fruit_x'] / self.width
        obs[3] = obs_raw['fruit_y'] / self.height

        return obs


def test_CatcherBase():
    render = True
    if render:
        pygame.init()
    game = CatcherBase(width=256, height=256, render=render)
    game.rng = np.random.RandomState(24)
    if render:
        game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        # action = np.random.choice(2)
        # game.step(action)
        game.step_old(dt)
        state = game.get_game_state()
        print(state)
        if render:
            pygame.display.update()


def test_Catcher():
    render = True
    game = Catcher(width=521, height=521)
    game.env_init({'random_seed': 42, 'render': render})
    obs = game.env_start()
    print(obs)

    while True:
        action = np.random.choice(2)
        print(action)
        reward_obs_term = game.env_step(action)
        time.sleep(0.5)
        print(reward_obs_term[0], reward_obs_term[1])


if __name__ == "__main__":
    # test_CatcherBase()
    test_Catcher()

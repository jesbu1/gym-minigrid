#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


class FourRoomsSkillsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, train: bool, skills: list, agent_pos=None, goal_pos=None, visualize=False):
        # TODO: Remember to make sure skills are already removed according to length range
        self._train = train
        self._skills = skills
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.goal_pos = None
        self._visualize=visualize  # set True for visualization (see manual_control_skills.py)
        grid_size = 19
        # for large fourrooms, change to (grid_size=38, max_steps=200)
        super().__init__(grid_size=grid_size, max_steps=100)
        self.action_space = spaces.Discrete(len(self._skills))
        self.observation_space = spaces.Box(
            # [agent_x, agent_y, agent_dir, goal_x, goal_y]
            low = np.array([0, 0, 0, 0, 0]),
            high = np.array([grid_size - 1, grid_size - 1, 3, grid_size - 1, grid_size - 1])
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def add_heat(self, search_path):
        self.grid.add_heat(search_path)

    def update_skills(self, skills):
        self._skills = skills
        self.action_space = spaces.Discrete(len(self._skills))

    def set_path(self, i, j, skill):
        if (i,j) == self.goal_pos or i == self.agent_pos[0] and j == self.agent_pos[1]:
            return
        self.grid.set_path(i,j,skill)

    def step(self, action, skill=None):
        total_reward = 0
        actual_action = self._skills[action]
        for primitive_action in actual_action:
            obs, reward, done, info = MiniGridEnv.step(self, primitive_action)
            total_reward += reward
            if done:
                break
        if self._visualize:
            return (obs, self.build_obs()), total_reward, done, info
        else:
            return self.build_obs(), total_reward, done, info

    def reset(self):
        # keep resetting MiniGrid until training_valid depending on train/test
        while True: 
            # Current position and direction of the agent
            self.agent_pos = None
            self.agent_dir = None

            # Generate a new random grid at the start of each episode
            # To keep the same grid for each episode, call env.seed() with
            # the same seed before calling env.reset()
            self._gen_grid(self.width, self.height)

            # These fields should be defined by _gen_grid
            assert self.agent_pos is not None
            assert self.agent_dir is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos)
            assert start_cell is None or start_cell.can_overlap()

            # Item picked up, being carried, initially nothing
            self.carrying = None

            # Step count since episode start
            self.step_count = 0
            
            # Generate goal
            self.goal_pos = self.grid.find_goal()
            
            if not self.training_valid() ^ self._train:  # XNOR
                obs = self.gen_obs()
                break

        # Return observation
        if self._visualize:
            return (obs, self.build_obs())
        else:
            return self.build_obs()

    # include here to prevent circular dependency
    def training_valid(self):
        agent_pos = self.agent_pos
        goal_pos = self.goal_pos
        agent_goal = (in_room(agent_pos), in_room(goal_pos))
        test_set = {
            (1, 3),
            (3, 1),
            (2, 4),
            (4, 2)
        }
        return agent_goal not in test_set

    def build_obs(self):
        return np.concatenate((self.agent_pos, [self.agent_dir], self.goal_pos), axis=0)


# include here to prevent circular dependency
def in_room(pos):
    # | 4 | 1 |
    # | 3 | 2 |
    w, h = pos
    if w < 9 and h < 9:
        return 4
    elif w < 9 and h > 9:
        return 3
    elif w > 9 and h < 9:
        return 1
    else:
        return 2


register(
    id='MiniGrid-FourRoomsSkills-v0',
    entry_point='gym_minigrid.envs:FourRoomsSkillsEnv'
)

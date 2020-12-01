import gym_minigrid
from gym_minigrid.wrappers import DirectionObsWrapper 
import gym
env = gym.make("MiniGrid-FourRooms-v0")
env = DirectionObsWrapper(env)
env.reset()
agent_pos = env.agent_pos
goal_pos = env.goal_position
agent_dir = env.agent_dir
print(agent_pos, goal_pos, agent_dir)
# If reward is nonzero, then we know that we have reached the goal. env is reset automatically.

# Heuristic: manhattan distance to goal assuming we go forward from current direction. Ignore walls for now, and
# remember to include the one timestep we need to make a turn if needed to go to goal.
import pdb; pdb.set_trace()
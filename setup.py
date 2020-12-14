from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.1',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    install_requires=[
        'gym==0.12.1',
        'numpy>=1.15.0',
        'dahuffman==0.4.1',
        'torch==1.4.0',
        'torchvision==0.5.0',
        'pathos',
        'absl-py'
    ]
)

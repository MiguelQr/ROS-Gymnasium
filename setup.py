from setuptools import setup
import os
from glob import glob

package_name = 'rosgym'

# Rest of your setup.py content...

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 
              f'{package_name}.tasks', 
              f'{package_name}.tasks.goToPose', 
              f'{package_name}.utils'],
    # ... other setup parameters ...
    entry_points={
        'console_scripts': [
            # ... other entry points ...
            'multi_agent_training = rosgym.scripts.multi_agent_training:main',
        ],
        'gymnasium.envs': [
            'MultiRobotNavEnv-v0=rosgym.tasks.goToPose.multi_nav_discrete:MultiRobotNavEnv',
        ],
    },
    # ... other setup parameters ...
)

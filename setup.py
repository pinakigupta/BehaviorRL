######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from setuptools import setup, find_packages

setup(
    name='urban-env',
    version='0.0.dev1',
    description='An environment for simulated Urban driving tasks',
    url='',
    author='Munir Jojo-VErge',
    author_email='munirjojoverge@yahoo.es.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='Urban autonomous driving simulation environment for reinforcement learning research and testing',
    packages=find_packages(exclude=['docs', 'scripts', 'tests']),
    install_requires=['gym', 'numpy', 'pygame', 'jupyter', 'matplotlib', 'pandas'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        'deploy': ['pytest-runner', 'sphinx<1.7.3', 'sphinx_rtd_theme']
    },
    entry_points={
        'console_scripts': [],
    },
)


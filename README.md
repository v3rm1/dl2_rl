# Reinforcement Learning: Augmenting Reward with Confidence Estimation

## Usage
Make sure you have the requirements to run the code:
```
pip install -r requirements.txt
```

To run enviroment with E-Greedy:
```
cd e_greedy
python3 pong.py
```

To run enviroment with Proximal Policy Optimization:
```
cd proximal_policy_optimization
python3 main.py
```

To run enviroment with Proximal Policy Optimization augmented with Confidence Estimation:
```
cd ppo_confidence
python3 main.py
```

## Introduction
There are a lot of reasons for wanting AI systems to be able to accurately estimate their own confidence, particularly in domains where failure incurs a real world cost. 

We have devised an unconventional method of including confidence estimation in a reinforcement learning system which we hypothesize will actually improve performance by forcing the system to develop a more informative model of the state-action space.

The method we propose is to construct a system which, instead of just outputting actions, gives action/confidence pairs, where the confidence is a number between 0 or 1 which represents the system's best guess as to the probability of the action is selected resulting in a successful outcome.

This confidence is then trained by altering the reward function such that the reward is a function of both the real value of the action selected in the environment (success or failure), as well as the confidence given. We want to reward the system for giving a high confidence when it selects actions correctly, and to punish the system for a high confidence when it selects actions poorly. 

This change in the reward function will vary the magnitude of punishment and reward, and thus change how the system learns. 

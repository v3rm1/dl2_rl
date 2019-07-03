from env import Env
from ppo import Agent
import numpy as np
import random

def downsample(state):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return state[::2, ::2, :]

def remove_color(state):
    """Convert all color (RGB is the third dimension in the image)"""
    return state[:, :, 0]

def remove_background(state):
    state[state == 144] = 0
    state[state == 109] = 0
    return state

def preprocess_observations(state):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = state[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 
    return processed_observation

def main(dic_agent_conf, dic_env_conf, dic_exp_conf, dic_path):
    env = Env(dic_env_conf)

    dic_agent_conf["ACTION_DIM"] = env.action_dim
    dic_agent_conf["STATE_DIM"] = env.state_dim

    agent = Agent(dic_agent_conf, dic_path, dic_env_conf)

    for cnt_episode in range(dic_exp_conf["TRAIN_ITERATIONS"]):
        s = env.reset()
        s = preprocess_observations(s)
        r_sum = 0
        for cnt_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):
            if cnt_episode > dic_exp_conf["TRAIN_ITERATIONS"] - 10:
                env.render()

            a = agent.choose_action(s)
            print(a)
            s_, r, done, _ = env.step(a)
            s_ = preprocess_observations(s_)
            r /= 100
            r_sum += r
            if done:
                r = -1

            agent.store_transition(s, a, s_, r, done)
            if (cnt_step+1) % dic_agent_conf["BATCH_SIZE"] == 0 and cnt_step != 0:
                agent.train_network()
            s = s_

            if done:
                break

            if cnt_step % 10 == 0:
                print("Episode:{}, step:{}, r_sum:{}".format(cnt_episode, cnt_step, r_sum))



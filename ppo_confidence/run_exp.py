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
        game_not_over = True
        count = 0
        
        histogram = np.zeros(agent.n_actions)
        
        while game_not_over:
            #env.render()
            #if cnt_episode > dic_exp_conf["TRAIN_ITERATIONS"] - 10:
                #env.render()

            if dic_agent_conf["USING_CONFIDENCE"]:
                #choose a confidence-action pair instead of just an action
                (a, c) = agent.choose_action(s)
                if count % 1000 == 0:
                    state = np.reshape(s, [-1, agent.dic_agent_conf["STATE_DIM"][0]])
                    print("A_dist: {}".format(agent.actor_network.predict_on_batch([state, agent.dummy_advantage, agent.dummy_old_prediction]).flatten()[:-1]))
                    print("Conf: ", c)
                    print("Valuation: ", agent.get_v(s))
            else:
                a = agent.choose_action(s)
                if count % 1000 == 0:
                    print("Valuation: ", agent.get_v(s))

            histogram[a] += 1
            s_, r, done, _ = env.step(a)
            s_ = preprocess_observations(s_)

            r_sum += r
            if done:
                game_not_over = False

            if dic_agent_conf["USING_CONFIDENCE"]:
                agent.store_transition(s, a, s_, r, done, c)
            else:
                agent.store_transition(s, a, s_, r, done)
            s = s_
            
            count = count + 1
        
        histogram = [int(h) for h in histogram]
        print("Hist: {}".format(histogram))
        dic_agent_conf["BATCH_SIZE"] = count
        print("Episode:{}, r_sum:{}".format(cnt_episode, r_sum))
        agent.train_network(cnt_episode)
    agent.save_model("savedModel")



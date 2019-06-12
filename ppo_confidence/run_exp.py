from env import Env
from ppo import Agent
import numpy as np

def main(dic_agent_conf, dic_env_conf, dic_exp_conf, dic_path):
    env = Env(dic_env_conf)

    dic_agent_conf["ACTION_DIM"] = env.action_dim
    dic_agent_conf["STATE_DIM"] = (env.state_dim, )

    agent = Agent(dic_agent_conf, dic_path, dic_env_conf)

    for cnt_episode in range(dic_exp_conf["TRAIN_ITERATIONS"]):
        s = env.reset()
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
                #if count % 1000 == 0:
                    #print("Action: ", a)
                    #print("Conf: ", c)
                    #print("Valuation: ", agent.get_v(s))
            else:
                a = agent.choose_action(s)
                #if count % 1000 == 0:
                    #print("Valuation: ", agent.get_v(s))

            histogram[a] += 1
            s_, r, done, _ = env.step(a)

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



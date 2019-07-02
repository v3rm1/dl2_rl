import random
import gym
import numpy as np
import cv2

class Pong:
    
#     https://github.com/llSourcell/pong_neural_network_live

# https://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/

    env = gym.make('PongNoFrameskip-v4')
    env.reset()
    actions = env.action_space.n
    init_action = 0
    init_state, init_reward, terminal, info = env.step(init_action)
    init_state = preprocess(init_action)
    
    def preprocess(state):
        state = cv2.cvtColor(cv2.resize(state, (84, 110)), cv2.COLOR_BGR2GRAY)
        state = state[26:110, :]
        ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(state, (84, 84, 1))

    def choose_action(action_prob):
        


    #
    def getPresentFrame(self):
        

        
        return image_data

    #update our screen
    def getNextFrame(self, action):
        
        #TODO: Add a way to update frame

        #record the total score
        self.tally = self.tally + score
        print("Tally is " + str(self.tally))
        #return the score and the surface data
        return [score, image_data]
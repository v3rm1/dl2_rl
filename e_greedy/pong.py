import gym
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, LeakyReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from collections import deque
import numpy as np
import random
from time import strftime
import pandas as pd


episodes = 500000
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        input_shape = (80,80, 1)
        num_classes = 6
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=2,input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(39, kernel_size=2))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return self.memory
    
    def act(self, state):
        state = state.reshape((-1, 80, 80, 1))
        if np.random.rand() <= self.epsilon:
            return random.randint(2, 3)
        act_values = self.model.predict(state)
        if act_values[0][2]>act_values[0][3]:
            return 2
        else:
            return 3
    
    def replay(self, batch_size):
        ret_eps = []
        ret_target = []
        ret_reward = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((-1, 80, 80, 1))
            next_state = next_state.reshape((-1, 80, 80, 1))
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            ret_eps.append(self.epsilon)
            ret_target.append(target)
            ret_reward.append(reward)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(self.epsilon)
        return ret_eps, ret_target, self.model, ret_reward



def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions=(80,80)):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 
    # processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

if __name__ == "__main__":
    action_batch = 1000
    
    # initialize gym environment and the agent
    curr_eps = []
    curr_reward = []
    curr_target = []
    e = 0

    env = gym.make('PongNoFrameskip-v4')
    agent = DQNAgent(env.observation_space, env.action_space.n)
    prev_states = None
    print(agent.epsilon)
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state, prev_states = preprocess_observations(state, prev_states)
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(50000):
            # turn this on if you want to render
            # env.render()
            print(time_t)
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            prev_states, next_state = preprocess_observations(next_state, prev_states)
            # Remember the previous state, action, reward, and done
            curr_mem = agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            
            if done:
                
                learn_file = "./e_greedy/memory/episode_"+str(e)+ "_"+strftime("%a_%d_%b_%y__%H%M%S")+".csv"
                mem_df = pd.DataFrame(columns=['Episode', 'Action', 'Reward', 'Done'])
                for i in range(0,len(curr_mem)):
                    mem_df = mem_df.append({'Episode':e, 'Action': curr_mem[i][1], 'Reward': curr_mem[i][2], 'Done': curr_mem[i][4]}, ignore_index=True)
                mem_df.to_csv(learn_file)
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))

                print(agent.epsilon)
                      
                break
        # train the agent with the experience of the episode
        curr_eps, curr_target, curr_model, curr_reward = agent.replay(action_batch)
        model_file = "./e_greedy/models/k_ep_"+str(e)+"_"+strftime("%a_%d_%b_%y__%H%M%S")+".h5"
        replay_data = "./e_greedy/replay/episode_"+str(e)+ "_"+strftime("%a_%d_%b_%y__%H%M%S")+".csv"
        

        replay_df = pd.DataFrame(columns=['Episode', 'Epsilon', 'Target', 'Reward'])
        for i in range(0,action_batch-1):
            replay_df = replay_df.append({'Episode': e, 'Epsilon': curr_eps[i], 'Target': curr_target[i], 'Reward': curr_reward[i]}, ignore_index=True)
        replay_df.to_csv(replay_data)
        curr_model.save(model_file)
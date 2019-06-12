import gym
import math
import memory
import model
import random
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.8
BATCH_SIZE = 50


#TODO: Add preproc and figure out why the array sizes do not match
class Game:
    """
    Defining a class for initializing and running the game
    """
    def __init__(self, sess, model, env, memory, max_epsilon, min_epsilon, decay, render_game=False):
        self._sess = sess
        self._model = model
        self._env = env
        self._memory = memory
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay = decay
        self._steps = 0
        self._stored_rewards = []
        self._stored_max_x = []
        self._epsilon = self._max_epsilon
        self._render = render_game

    
    def run(self):
        state = self._env.reset()
        total_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)
            print("Current action: {}\nCurrent Reward: {}".format(action, reward))
            # print(next_state[0])
            # if next_state[0] >= 0.1:
            #     reward += 10
            # elif next_state[0] >= 0.25:
            #     reward += 20
            # elif next_state[0] >= 0.5:
            #     reward += 100

            # if next_state[0] > max_x:
            #     max_x = next_state[0]
            
            if done:
                next_state = None
            
            self._memory.add_sample((state, action, reward, next_state))
            # self._replay()

            self._steps += 1
            self._epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)
            print("Updated epsilon: {}".format(self._epsilon))
            if self._steps > 50:
                self._replay()

            state = next_state
            total_reward += reward

            if done:
                self._stored_rewards.append(total_reward)
                self._stored_max_x.append(max_x)
            
            print("Step {}:\n Total Reward: {}, Epsilon: {}".format(self._steps, total_reward, self._epsilon))
            
    def _choose_action(self, state):
        i = random.random()
        print("Random.random: ", i)
        if i > self._epsilon:
            return random.randint(0, self._model.action_count - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))
        
    
    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = []
        next_states = []
        for i in range(self._model.batch_size):
            states.append(batch[i][0])
            if batch[i][3] is None:
                next_states.append(np.zeros(self._model.state_count))
            else:
                next_states.append(batch[i][3])
        
        # states = np.array(batch_val[0][0] for batch_val in batch)
        # states = self._memory.sample(self._model.batch_size)[0][0]
        # next_states = np.array([(np.zeros(self._model.state_count) if batch_val[3] is None else batch_val[3] for batch_val in batch)])
        q_value = self._model.predict_batch(np.reshape(states, (len(batch), self._model.state_count)), self._sess)
        q_value_pred = self._model.predict_batch(np.reshape(next_states, (len(batch), self._model.state_count)), self._sess)
        x = np.zeros((len(batch), self._model.state_count))
        y = np.zeros((len(batch), self._model.action_count))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q_value = q_value[i]
            if next_state is None:
                current_q_value[action] = reward
            else:
                current_q_value[action] = reward + GAMMA * np.amax(q_value_pred[i])
                print("Q Value Predicted: ", q_value_pred[i])
                print("Current Q Value Action: ", current_q_value[action])
            
            x[i] = np.reshape(state, (1, self._model.state_count))
            y[i] = current_q_value
        
        self._model.train_batch(self._sess, x, y)

        @property
        def stored_rewards(self):
            return self._stored_rewards
        
        @property
        def stored_max_x(self):
            return self._stored_max_x



if __name__=="__main__":
    env_name = "PongNoFrameskip-v4"
    env = gym.make(env_name)

    num_states = (210*160*3)
    num_actions = env.action_space.n

    model = model.Model(num_states, num_actions, BATCH_SIZE)
    mem = memory.Memory(50000)

    with tf.Session() as sess:
        sess.run(model.var_init)
        game = Game(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        num_episodes = 500
        count = 0
        while count < num_episodes:
            if count % 10 == 0:
                print('Episode {} of {}'.format(count+1, num_episodes))
            game.run()
            count += 1
        plt.plot(game.reward_store)
        plt.show()
        plt.close("all")
        plt.plot(game.max_x_store)
        plt.show()

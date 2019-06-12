from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, RMSprop, Adadelta
import os
from keras.layers import Input, Dense, LeakyReLU
from keras import initializers
import keras.backend as K
import time
from copy import deepcopy
import numpy as np
import math


class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.batch_conf = []

    def store(self, s, a, s_, r, done, c=None):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)
        if not c is None:
            self.batch_conf.append(c)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()

    @property
    def cnt_samples(self):
        return len(self.batch_s)


class Agent:
    def __init__(self, dic_agent_conf, dic_path, dic_env_conf):
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.dic_env_conf = dic_env_conf

        self.n_actions = self.dic_agent_conf["ACTION_DIM"]

        self.actor_network = self._build_actor_network()
        self.actor_old_network = self.build_network_from_copy(self.actor_network)

        self.critic_network = self._build_critic_network()
        
        self.confidence_network = self._build_confidence_network()

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.n_actions))

        self.memory = Memory()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"
        state = np.reshape(state, [-1, self.dic_agent_conf["STATE_DIM"][0]])

        prob = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        action = np.random.choice(self.n_actions, p=prob)
        
        if self.dic_agent_conf["USING_CONFIDENCE"]:
            action_vector = np.zeros((1, self.n_actions))
            action_vector[0][action] = 1
            action_state = np.concatenate((action_vector, state), axis = 1)
            conf = self.confidence_network.predict_on_batch(action_state)
			
            return (action, conf[0][0])
        else:
            return action
            
    def get_reward_multiplier(self, win_lose, c):
        if c < self.dic_agent_conf["MINIMUM_CONFIDENCE"]:
            c = self.dic_agent_conf["MINIMUM_CONFIDENCE"]
        if c > self.dic_agent_conf["MAXIMUM_CONFIDENCE"]:
            c = self.dic_agent_conf["MAXIMUM_CONFIDENCE"]
        if win_lose == 1:
            return math.sqrt(c) / math.sqrt(0.5) #function normalized about confidence of 50%
        if win_lose == -1:
            return (math.sqrt(c) + (math.log((1 - math.sqrt(c))/(1 + math.sqrt(c))))/2) * -1 / math.sqrt(0.5) 
        print("ERROR: get_reward_multiplier got an incorrect value")
        exit()

    def train_network(self):
        n = self.memory.cnt_samples
        discounted_r = []
        batch_win_lose = []
        
        #make sure that the game ended
        assert(self.memory.batch_done[-1])        
        #make sure that the last state in the game has a reward of either 1 or -1
        assert(self.memory.batch_r[-1] != 0)
        
        if self.dic_agent_conf["USING_CONFIDENCE"]:
            win_lose = 0
            #loops backwards over memory
            for i in range(len(self.memory.batch_r)-1, -1, -1):
                r = self.memory.batch_r[i]
                if r != 0:
                    win_lose = r
                    v = r
                else:
                    v = v * self.dic_agent_conf["GAMMA"]
                discounted_r.append(v * self.get_reward_multiplier(win_lose, self.memory.batch_conf[i]))
                batch_win_lose.append((win_lose+1.04)/2.08) #zero is a loss, one is a win (using soft targets here (0.02, 0.98)
        else:
            win_lose = 0
            #loops backwards over memory
            for i in range(len(self.memory.batch_r)-1, -1, -1):
                r = self.memory.batch_r[i]
                if r != 0:
                    win_lose = r
                    v = r
                else:
                    v = v * self.dic_agent_conf["GAMMA"]
                discounted_r.append(v)
        discounted_r.reverse()
        batch_win_lose.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(np.reshape(self.memory.batch_s, (self.dic_agent_conf["BATCH_SIZE"], self.dic_agent_conf["STATE_DIM"][0]))), \
                     np.vstack(self.memory.batch_a), \
                     np.vstack(discounted_r)
        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        
        #batch_advantage = np.full(batch_advantage.shape, -1)
        
        batch_old_prediction = self.get_old_prediction(batch_s)

        batch_a_final = np.zeros(shape=(len(batch_a), self.n_actions))
        batch_a_final[:, batch_a.flatten()] = 1
        
        if self.dic_agent_conf["USING_CONFIDENCE"]:
            batch_a_s = np.concatenate((batch_a_final, batch_s), axis=1)
            self.confidence_network.fit(x=batch_a_s, y=batch_win_lose, epochs = 2, verbose=0)			
        
        # print("TRAINING ACTOR NETWORK WITH THE FOLLOWING PARAMETERS:\nbatch_s:{}\nbatch_advantage:{}\nbatch_old_prediction:{}\nbatch_a_final:{}".format(batch_s.shape, batch_advantage.shape, batch_old_prediction.shape, batch_a_final.shape))
        self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.critic_network.fit(x=batch_s, y=batch_discounted_r, epochs=2, verbose=0)
        self.memory.clear()
        self.update_target_network()

    def get_old_prediction(self, s):
        s = np.reshape(s, (-1, self.dic_agent_conf["STATE_DIM"][0]))
        return self.actor_old_network.predict_on_batch(s)

    def store_transition(self, s, a, s_, r, done, c=None):
        if self.dic_agent_conf["USING_CONFIDENCE"]: 
            self.memory.store(s, a, s_, r, done, c)
        else:
            self.memory.store(s, a, s_, r, done)

    def get_v(self, s):
        s = np.reshape(s, (-1, self.dic_agent_conf["STATE_DIM"][0]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def save_model(self, file_name):
        self.actor_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5" % file_name))
        self.critic_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5" % file_name))

    def load_model(self):
        self.actor_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5")
        self.critic_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5")
        self.actor_old_network = deepcopy(self.actor_network)

    def _build_actor_network(self):

        state = Input(shape=self.dic_agent_conf["STATE_DIM"], name="state")
        # print("BUILD ACTOR NETWORK: STATE", state.shape)

        advantage = Input(shape=(1, ), name="Advantage")
        old_prediction = Input(shape=(self.n_actions,), name="Old_Prediction")

        shared_hidden = self._shared_network_structure(state)

        action_dim = self.dic_agent_conf["ACTION_DIM"]

        policy = Dense(action_dim, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="softmax", name="actor_output_layer")(shared_hidden)

        actor_network = Model(inputs=[state, advantage, old_prediction], outputs=policy)

        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            actor_network.compile(optimizer=Adadelta(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                  loss=self.proximal_policy_optimization_loss(
                                    advantage=advantage, old_prediction=old_prediction,
                                  ))
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            actor_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        print("=== Build Actor Network ===")
        actor_network.summary()

        time.sleep(1.0)
        return actor_network

    def update_target_network(self):
        alpha = self.dic_agent_conf["TARGET_UPDATE_ALPHA"]
        self.actor_old_network.set_weights(alpha*np.array(self.actor_network.get_weights())
                                           + (1-alpha)*np.array(self.actor_old_network.get_weights()))
                                        
    def _build_confidence_network(self):
        state_action = Input(shape= (self.dic_agent_conf["STATE_DIM"][0] + self.n_actions, ), name="state_and_action")
		
        shared_hidden = self._shared_network_structure(state_action)
		
        q = Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="sigmoid", name="confidence_output_layer")(shared_hidden)
		
        confidence_network = Model(inputs=state_action, outputs=q)
		
        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            confidence_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]), 
                                    loss=self.dic_agent_conf["CRITIC_LOSS"])
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            confidence_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                    loss=self.dic_agent_conf["CRITIC_LOSS"])
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            confidence_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                    loss=self.dic_agent_conf["CRITIC_LOSS"])
                                   
        print("=== Build Confidence Network ===")
        confidence_network.summary()

        time.sleep(1.0)
        return confidence_network

    def _build_critic_network(self):
        state = Input(shape=self.dic_agent_conf["STATE_DIM"], name="state")
        # print("BUILD CRITIC NETWORK: STATE", state.shape)

        shared_hidden = self._shared_network_structure(state)

        if self.dic_env_conf["POSITIVE_REWARD"]:
            q = Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01), 
                    bias_initializer=initializers.Constant(0.1), activation="relu", name="critic_output_layer")(shared_hidden)
        else:
            q = Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), name="critic_output_layer")(shared_hidden)

        critic_network = Model(inputs=state, outputs=q)

        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            critic_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            critic_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            critic_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        print("=== Build Critic Network ===")
        critic_network.summary()

        time.sleep(1.0)
        return critic_network

    def build_network_from_copy(self, actor_network):
        network_structure = actor_network.to_json()
        network_weights = actor_network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]), loss="mse")
        return network

    def _shared_network_structure(self, state_features):
        dense_d = self.dic_agent_conf["D_DENSE"]
        hidden1 = Dense(dense_d, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="linear", name="hidden_shared_1")(state_features)
        hidden1_leaky = LeakyReLU(alpha=.1)(hidden1)
        hidden2 = Dense(dense_d, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="relu", name="hidden_shared_2")(hidden1_leaky)
        hidden2_leaky = LeakyReLU(alpha=.1)(hidden2)
        return hidden2_leaky

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        loss_clipping = self.dic_agent_conf["CLIPPING_LOSS_RATIO"]
        entropy_loss = self.dic_agent_conf["ENTROPY_LOSS_RATIO"]

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage)) # + entropy_loss * (
                           #prob * K.log(prob + 1e-10)))

        return loss

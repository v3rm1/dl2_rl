from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, RMSprop, Adadelta
import os
from keras.layers import Input, Dense, LeakyReLU, Concatenate, concatenate, Conv2D, Flatten, MaxPool2D
from keras import initializers
import keras.backend as K
import time
from copy import deepcopy
import numpy as np
import math
import random


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

        if self.dic_agent_conf["USING_CONFIDENCE"]:
            self.actor_network = self._build_actor_network_confidence()
        else:
            self.actor_network = self._build_actor_network()
        
        self.critic_network = self._build_critic_network()
        self.actor_old_network = self.build_network_from_copy(self.actor_network)

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.n_actions))

        self.memory = Memory()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"
        state = state.reshape((-1, 80, 80, 1))
        output = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()

        if self.dic_agent_conf["USING_CONFIDENCE"]:
            prob = output[:-1]
            action = np.random.choice(self.n_actions, p=prob)
            conf = output[-1]
            #if conf < self.dic_agent_conf["MINIMUM_CONFIDENCE"]:
                #action = random.randint(0,self.n_actions-1)
            return (action, conf)
        else:
            action = np.random.choice(self.n_actions, p=output)
            return action
    
    def get_confidence_multiplier(self, conf, win_lose):
        if win_lose == 1:
            return math.sqrt(conf)
        if win_lose == -1:
            return -1*(math.sqrt(conf) + 0.5*math.log((1 - math.sqrt(conf))/(1 + math.sqrt(conf))))
            
    def train_network(self, episode):
        n = self.memory.cnt_samples
        discounted_r = []
        batch_win_lose = []
        
        #make sure that the game ended
        assert(self.memory.batch_done[-1])        
        #make sure that the last state in the game has a reward of either 1 or -1
        assert(self.memory.batch_r[-1] != 0)
        
        win_lose = 0
        #loops backwards over memory
        for i in range(len(self.memory.batch_r)-1, -1, -1):
            r = self.memory.batch_r[i]
            if r != 0:
                win_lose = r
                v = r
            else:
                v = v * self.dic_agent_conf["GAMMA"]
            if self.dic_agent_conf["USING_CONFIDENCE"]: 
                discounted_r.append(v*self.get_confidence_multiplier(self.memory.batch_conf[i], win_lose))
            else:
                discounted_r.append(v)
            batch_win_lose.append([win_lose])


        discounted_r.reverse()
        batch_win_lose.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(np.reshape(self.memory.batch_s, (self.dic_agent_conf["BATCH_SIZE"], -1, 80, 80, 1))), \
                     np.vstack(self.memory.batch_a), \
                     np.vstack(discounted_r)
        
        batch_old_prediction = self.get_old_prediction(batch_s)

        batch_a_final = np.eye(self.n_actions)[self.memory.batch_a]
        
        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        
        if self.dic_agent_conf["USING_CONFIDENCE"]:
            batch_old_prediction = batch_old_prediction[:,:-1]
            output = np.concatenate((batch_a_final, batch_win_lose), axis = 1)
        else:
            output = batch_a_final
        self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=output, epochs = 2, verbose=1)
        self.critic_network.fit(x=batch_s, y=batch_discounted_r, epochs=5, verbose=1)

        self.memory.clear()
        self.update_target_network()

    def get_old_prediction(self, s):
        s = s.reshape((-1, 80, 80, 1))
        return self.actor_old_network.predict_on_batch(s)

    def store_transition(self, s, a, s_, r, done, c=None):
        if self.dic_agent_conf["USING_CONFIDENCE"]: 
            self.memory.store(s, a, s_, r, done, c)
        else:
            self.memory.store(s, a, s_, r, done)

    def get_v(self, s):
        s = s.reshape((-1, 80, 80, 1))
        v = self.critic_network.predict_on_batch(s)
        return v

    def save_model(self, file_name):
        self.actor_network.save(os.path.join("./models/%s_actor_network.h5" % file_name))
        self.critic_network.save(os.path.join("./models/%s_critic_network.h5" % file_name))

    def load_model(self):
        self.actor_network = load_model("./models/%s_actor_network.h5")
        self.critic_network = load_model("./models/%s_critic_network.h5")
        self.actor_old_network = deepcopy(self.actor_network)

    def _build_actor_network_confidence(self):

        state = Input(shape=self.dic_agent_conf["STATE_DIM"], name="state")
        # print("BUILD ACTOR NETWORK: STATE", state.shape)

        advantage = Input(shape=(1, ), name="Advantage")
        old_prediction = Input(shape=(self.n_actions,), name="Old_Prediction")

        shared_hidden = self._shared_network_structure(state)

        action_dim = self.dic_agent_conf["ACTION_DIM"]
        
        act_policy = Dense(action_dim, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="softmax", name="actor_output_layer")(shared_hidden)
                    
        act_plus_shared = Concatenate()([act_policy, shared_hidden])
                    
        conf_policy = Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="sigmoid", name="confidence_output_layer")(act_plus_shared)
        
                    
        policy = Concatenate()([act_policy, conf_policy])
        actor_network = Model(inputs=[state, advantage, old_prediction], outputs=policy)

        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                  loss=self.confidence_loss(
                                    advantage=advantage, old_prediction=old_prediction,
                                  ))
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            actor_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        print("=== Build Actor Network ===")
        actor_network.summary()

        #time.sleep(1.0)
        return actor_network
        
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
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
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

        #time.sleep(1.0)
        return actor_network

    def update_target_network(self):
        alpha = self.dic_agent_conf["TARGET_UPDATE_ALPHA"]
        self.actor_old_network.set_weights(alpha*np.array(self.actor_network.get_weights())
                                           + (1-alpha)*np.array(self.actor_old_network.get_weights()))

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
        conv1 = Conv2D(16, kernel_size=2, name="conv_shared_1", input_shape=(80, 80, 1))(state_features)
        conv1_leaky = LeakyReLU(alpha=0.1)(conv1)
        conv2 = Conv2D(32, kernel_size=2, name="conv_shared_2")(conv1_leaky)
        conv2_leaky = LeakyReLU(alpha=0.1)(conv2)
        flatten1 = Flatten()(conv2_leaky)
        hidden1 = Dense(dense_d, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="linear", name="hidden_shared_1")(flatten1)
        hidden1_leaky = LeakyReLU(alpha=.1)(hidden1)
        hidden2 = Dense(dense_d, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Constant(0.1), activation="linear", name="hidden_shared_2")(hidden1_leaky)
        hidden2_leaky = LeakyReLU(alpha=.1)(hidden2)
        return hidden2_leaky
        
    def confidence_loss(self, advantage, old_prediction):
        loss_clipping = self.dic_agent_conf["CLIPPING_LOSS_RATIO"]
        entropy_loss = self.dic_agent_conf["ENTROPY_LOSS_RATIO"]
        min_conf = self.dic_agent_conf["MINIMUM_CONFIDENCE"]
        max_conf = self.dic_agent_conf["MAXIMUM_CONFIDENCE"]
        
        def loss(y_true, y_pred):
            act_true = y_true[:,:-1]
            act_pred = y_pred[:,:-1]
            conf_pred = K.clip(y_pred[:,-1], min_value=min_conf, max_value=max_conf)
            
            win_lose = y_true[:,-1]
            win = (win_lose+1)/2
            lose = (1-win_lose)/2
            
            lose_loss = lose*(K.sqrt(conf_pred) + K.log((1-K.sqrt(conf_pred))/(1+K.sqrt(conf_pred))))
            win_loss = win*(K.sqrt(conf_pred))
            c_loss = -1*(lose_loss + win_loss)
            
            prob = act_true * act_pred
            old_prob = act_true * old_prediction
            r = prob / (old_prob + 1e-10)
            loss_e = K.mean(entropy_loss * (act_pred * K.log(act_pred)))
            loss_p = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage))
            
            return K.mean(loss_e + loss_p + c_loss)
        
        return loss            
            

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        loss_clipping = self.dic_agent_conf["CLIPPING_LOSS_RATIO"]
        entropy_loss = self.dic_agent_conf["ENTROPY_LOSS_RATIO"]

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            loss_e = K.mean(entropy_loss * (y_pred * K.log(y_pred)))
            loss_p = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage))
            return K.mean(loss_e + loss_p)

        return loss

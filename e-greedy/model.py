import tensorflow as tf
import numpy as np

# TODO: Add method comments and verify if the neural network structure is valid
class Model:
    """
    Defining the model class
    """
    def __init__(self, state_count, action_count, batch_size):
        self._state_count = state_count
        self._action_count = action_count
        self._batch_size = batch_size

        # Placeholders for state and action space
        self._states = None
        self._actions = None
        
        # Output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._state_count], dtype=tf.float32)
        self._q_value = tf.placeholder(shape=[None, self._action_count], dtype=tf.float32)
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._action_count)
        loss = tf.losses.mean_squared_error(self._q_value, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: np.reshape(state, (1, self._state_count))})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_value: y_batch})

    
    @property
    def state_count(self):
        return self._state_count

    @property
    def action_count(self):
        return self._action_count

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
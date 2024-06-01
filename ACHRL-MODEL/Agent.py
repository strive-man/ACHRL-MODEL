import tensorflow as tf


class AgentNetwork(object):
    '''
    agent network
    use the state to sample the action 状态采样采取的动作
    '''

    def __init__(self, sess, args):
        self.global_step = tf.Variable(0, trainable=False, name="AgentStep")
        self.sess = sess
        self.lr = args.agent_pretrain_lr
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.lr, self.global_step, 10000, 0.95,
                                                                  staircase=True)
        self.tau = args.agent_pretrain_tau
        self.high_state_size = args.high_state_size
        self.low_state_size = args.low_state_size
        self.weight_size = args.agent_weight_size
        self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
        self.num_other_variables = len(tf.compat.v1.trainable_variables())
        '''
        high-level network
        '''
        self.high_input_state, self.high_prob = self.create_agent_network("Activate/high", self.high_state_size)
        self.high_network_params = tf.compat.v1.trainable_variables()[self.num_other_variables:]
        self.target_high_input_state, self.target_high_prob = self.create_agent_network("Target/high",
                                                                                        self.high_state_size)
        self.target_high_network_params = tf.compat.v1.trainable_variables()[
                                          self.num_other_variables + len(self.high_network_params):]
        '''
        low-level network
        '''
        self.low_input_state, self.low_prob = self.create_agent_network("Activate/low", self.low_state_size)
        self.low_network_params = tf.compat.v1.trainable_variables()[
                                  self.num_other_variables + len(self.high_network_params) + len(
                                      self.target_high_network_params):]

        self.target_low_input_state, self.target_low_prob = self.create_agent_network("Target/low", self.low_state_size)
        self.target_low_network_params = tf.compat.v1.trainable_variables()[
                                         self.num_other_variables + len(self.high_network_params) + len(
                                             self.target_high_network_params) + len(self.low_network_params):]
        print(self.target_low_network_params)
        self.update_target_high_network_params = \
            [self.target_high_network_params[i].assign( \
                tf.multiply(self.high_network_params[i], self.tau) + \
                tf.multiply(self.target_high_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_high_network_params))]

        self.update_target_low_network_params = \
            [self.target_low_network_params[i].assign( \
                tf.multiply(self.low_network_params[i], self.tau) + \
                tf.multiply(self.target_low_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_low_network_params))]

        self.assign_active_high_network_params = \
            [self.high_network_params[i].assign( \
                self.target_high_network_params[i]) for i in range(len(self.high_network_params))]

        self.assign_active_low_network_params = \
            [self.low_network_params[i].assign( \
                self.target_low_network_params[i]) for i in range(len(self.low_network_params))]

        self.td_error = tf.compat.v1.placeholder(shape=[None, 256], dtype=tf.float32)
        self.high_action_holder = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.high_pi = self.high_action_holder * self.target_high_prob + (1 - self.high_action_holder) * (
                1 - self.target_high_prob)
        self.high_loss = -tf.reduce_sum(input_tensor=tf.math.log(self.high_pi) * self.td_error)
        self.high_gradients = tf.gradients(ys=self.high_loss, xs=self.target_high_network_params)
        self.low_action_holder = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.td_error_l = tf.compat.v1.placeholder(shape=[None, 25], dtype=tf.float32)
        self.low_pi = self.low_action_holder * self.target_low_prob + (1 - self.low_action_holder) * (
                1 - self.target_low_prob)
        self.low_loss = -tf.reduce_sum(input_tensor=tf.math.log(self.low_pi) * self.td_error_l)
        self.low_gradients = tf.gradients(ys=self.low_loss, xs=self.target_low_network_params)
        self.high_grads = [tf.compat.v1.placeholder(tf.float32, [self.high_state_size, 1]),
                           tf.compat.v1.placeholder(tf.float32, [1, 1])]
        self.low_grads = [tf.compat.v1.placeholder(tf.float32, [self.low_state_size, 1]),
                          tf.compat.v1.placeholder(tf.float32, [1, 1])]
        self.high_gradient_holders = []
        for idx, var in enumerate(self.high_network_params):
            placeholder = tf.compat.v1.placeholder(tf.float32, name=str(idx) + '_holder')
            self.high_gradient_holders.append(placeholder)
        self.high_optimize = self.optimizer.apply_gradients(zip(self.high_gradient_holders, self.high_network_params),
                                                            global_step=self.global_step)
        self.low_gradient_holders = []
        for idx, var in enumerate(self.low_network_params):
            placeholder = tf.compat.v1.placeholder(tf.float32, name=str(idx) + '_holder')
            self.low_gradient_holders.append(placeholder)
        self.low_optimize = self.optimizer.apply_gradients(zip(self.low_gradient_holders, self.low_network_params),
                                                           global_step=self.global_step)

    def udpate_tau(self, tau):
        self.tau = tau

    def update_lr(self, lr):
        self.lr = lr

    def create_agent_network(self, scope, state_size):
        with tf.compat.v1.name_scope(scope):
            input_state = tf.compat.v1.placeholder(shape=[None, state_size], dtype=tf.float32)
            embedding_size = state_size
            weight_size = self.weight_size
            W = tf.Variable(tf.random.truncated_normal(shape=[embedding_size, weight_size], mean=0.0,
                                                       stddev=tf.sqrt(
                                                           tf.compat.v1.div(2.0, weight_size + embedding_size))),
                            name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            b = tf.Variable(tf.constant(0, shape=[1, weight_size], dtype=tf.float32), name='Bias_for_MLP',
                            dtype=tf.float32, trainable=True)
            h = tf.Variable(
                tf.random.truncated_normal(shape=[weight_size, 1], mean=0.0,
                                           stddev=tf.sqrt(tf.compat.v1.div(2.0, weight_size))),
                name='H_for_MLP', dtype=tf.float32, trainable=True)
            MLP_output = tf.matmul(input_state, W) + b  # (b, e) * (e, w) + (1, w)
            MLP_output = tf.nn.relu(MLP_output)
            prob = tf.nn.sigmoid(
                tf.reduce_sum(input_tensor=tf.matmul(MLP_output, h), axis=1))  # (b, w) * (w,1 ) => (b, 1)
            prob = tf.clip_by_value(prob, 1e-5, 1 - 1e-5)
            return input_state, prob

    # 梯度缓存
    def init_high_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_high_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_high(self, high_gradbuffer, high_grads):
        for index, grad in enumerate(high_grads):
            high_gradbuffer[index] += grad
        feed_dict = dict(zip(self.high_gradient_holders, high_gradbuffer))
        self.sess.run(self.high_optimize, feed_dict=feed_dict)

    def init_low_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_low_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_low(self, low_gradbuffer, low_grads):
        for index, grad in enumerate(low_grads):
            low_gradbuffer[index] += grad
        feed_dict = dict(zip(self.low_gradient_holders, low_gradbuffer))
        self.sess.run(self.low_optimize, feed_dict=feed_dict)

    def predict_high_target(self, high_state):
        return self.sess.run(self.target_high_prob, feed_dict={
            self.target_high_input_state: high_state})

    def predict_low_target(self, low_state):
        return self.sess.run(self.target_low_prob, feed_dict={
            self.target_low_input_state: low_state})

    def get_high_gradient(self, high_state, td_error, high_action):
        return self.sess.run(self.high_gradients, feed_dict={
            self.target_high_input_state: high_state,
            self.td_error: td_error,
            self.high_action_holder: high_action})

    def get_low_gradient(self, low_state, td_error_l, low_action):
        return self.sess.run(self.low_gradients, feed_dict={
            self.target_low_input_state: low_state,
            self.td_error_l: td_error_l,
            self.low_action_holder: low_action})

    def update_target_high_network(self):
        self.sess.run(self.update_target_high_network_params)

    def update_target_low_network(self):
        self.sess.run(self.update_target_low_network_params)

    def assign_active_high_network(self):
        self.sess.run(self.assign_active_high_network_params)

    def assign_active_low_network(self):
        self.sess.run(self.assign_active_low_network_params)


class Critic_l(object):
    def __init__(self, sess, lr=0.0007):
        self.sess = sess
        self.s = tf.compat.v1.placeholder(tf.float32, [None, 34], "state_l")
        self.v_ = tf.compat.v1.placeholder(tf.float32, [None, 1], "v_next_l")
        self.r = tf.compat.v1.placeholder(tf.float32, None, 'r_l')

        with tf.compat.v1.variable_scope('Critic_l'):
            l1 = tf.compat.v1.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.compat.v1.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        with tf.compat.v1.variable_scope('squared_TD_error_l'):
            self.td_error = (self.r + (0.9 * self.v_ - self.v))
            # self.td_error = (self.r + (0.98 * self.v_ - self.v))
            self.loss = tf.square(self.td_error)

        with tf.compat.v1.variable_scope('train_l'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error = self.sess.run([self.td_error, self.train_op],
                                 {self.s: s, self.v_: v_, self.r: r})
        return td_error


class Critic(object):
    def __init__(self, sess, lr=0.0005):
        self.sess = sess
        self.s = tf.compat.v1.placeholder(tf.float32, [None, 18], "state")
        self.v_ = tf.compat.v1.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.compat.v1.placeholder(tf.float32, None, 'r')

        with tf.compat.v1.variable_scope('Critic'):
            l1 = tf.compat.v1.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.compat.v1.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.compat.v1.variable_scope('squared_TD_error'):
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.compat.v1.variable_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error = self.sess.run([self.td_error, self.train_op],
                                 {self.s: s, self.v_: v_, self.r: r})
        return td_error

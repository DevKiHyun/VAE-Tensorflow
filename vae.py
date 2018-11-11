import tensorflow as tf

class VAE:
    def __init__(self, config):
        self.weights = {}
        self.biases = {}
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.z_dim = config.z_dim
        self.n_hidden_list = [512, 256, 128]

    def _nn_layer(self, inputs, n_hidden, initializer_type='He', add_bias=False, name=None, activation=None, stddev=None):
        inputs_shape = inputs.get_shape().as_list()
        weights_size = [inputs_shape[-1], n_hidden]

        layer_weights = self._get_weights(weights_size, initializer_type, name, stddev)

        hidden_layer = tf.matmul(inputs, layer_weights)

        if add_bias == True:
            layer_bias = self._get_bias(n_hidden, name)

            hidden_layer = tf.add(hidden_layer, layer_bias)

        if activation != None:
            hidden_layer = activation(hidden_layer)

        return hidden_layer

    def _get_weights(self, weights_size, initializer_type, name=None, stddev=None):
        name = name + "_weights"

        if stddev != None:
            initializer = tf.random_normal
        else:
            if initializer_type == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer()
            elif initializer_type == 'Xavier':
                initializer = tf.contrib.layers.xavier_initializer(uniform=False)

        layer_weights = tf.Variable(initializer(shape=weights_size), name=name, trainable=True)
        #layer_weights = tf.get_variable(name=name, shape=weights_size, initializer=initializer)

        return layer_weights

    def _get_bias(self, bias_size, name=None):
        name = name + "_bias"

        layer_bias = tf.Variable(tf.zeros([bias_size]), name=name, trainable=True)

        return layer_bias

    def reconstruction(self):
        # Expectation == Monte-Carlo-estimation
        epsilon = 1e-10
        recons = tf.reduce_sum(self.X*tf.log(epsilon + self.output) + (1 - self.X)*tf.log(epsilon + 1 - self.output), axis=1)
        recons = - recons
        recons = tf.reduce_mean(recons)

        return recons

    def regularization(self):
        # KL-divergence
        regulars = 0.5 * tf.reduce_sum(-1 - tf.log(1e-8 + tf.square(self.sigma)) + tf.square(self.mu) + tf.square(self.sigma), axis=1)
        regulars = tf.reduce_mean(regulars)

        return regulars

    def build(self):
        # Encoder
        self.z_encoder_layer = self.X
        count = 0
        for n_hidden in self.n_hidden_list:
            variables_name = 'z_{}dim_encoder_{}'.format(self.z_dim, count)
            self.z_encoder_layer = self._nn_layer(self.z_encoder_layer, n_hidden=n_hidden, activation=tf.nn.elu, name=variables_name)
            count += 1

        # mu and sigma are parameters of Gaussian distribution Z
        variables_name = 'z_{}dim_mu'.format(self.z_dim)
        self.mu = self._nn_layer(self.z_encoder_layer, n_hidden=self.z_dim, name=variables_name)

        variables_names = ['z_{}dim_sigma'.format(self.z_dim), 'z_{}dim_softplus'.format(self.z_dim)]
        self.sigma = self._nn_layer(self.z_encoder_layer, n_hidden=self.z_dim, name=variables_names[0])
        self.sigma = 1e-6 + tf.nn.softplus(self.sigma, name=variables_names[1])

        # noise for sample z
        self.noise = tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # sampled z
        self.sampled_z = tf.add(self.mu, self.sigma*self.noise)

        # Decoder
        self.z_decoder_layer = self.sampled_z
        count = 0
        for n_hidden in reversed(self.n_hidden_list):
            variables_name = 'z_{}dim_decoder_{}'.format(self.z_dim, count)
            self.z_decoder_layer = self._nn_layer(self.z_decoder_layer, n_hidden=n_hidden, activation=tf.nn.elu, name=variables_name)
            count += 1

        X_size = self.X.get_shape().as_list()[-1]
        self.output = self._nn_layer(self.z_decoder_layer, n_hidden=X_size, activation=tf.nn.sigmoid, name="z_{}dim_reconstruction".format(self.z_dim))

    def optimize(self, config):
        self.learning_rate = config.learning_rate
        self.beta_1 = config.beta_1
        self.beta_2 = config.beta_2
        self.epsilon = config.epsilon

        # LOSS = - ELBO
        self.recons = self.reconstruction()
        self.regular = self.regularization()
        self.cost = self.recons + self.regular
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1,
                                                beta2=self.beta_2, epsilon=self.epsilon).minimize(self.cost)
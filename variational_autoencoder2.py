import tensorflow as tf
import numpy
import pickle
import hdfs
import functional

class model(tf.keras.Model):
    '''Variational autoencoder class
    '''
    def __init__(self, n_inputs, encoding_feature_sizes, decoding_feature_sizes, dropOutRate=0.1,\
        scaler=None, feature_list=None, **kwargs):
        super(model, self).__init__(**kwargs)
        '''
        Arguments:
        n_inputs: Number of input features
        encoding_feature_sizes: List of integers. Length is the number of layers and the values 
        decoding_feature_sizes: List of integers. Length is the number of layers and the values 
        dropOutRate: Dropout rate. Defaults to 0.1
        scaler: Trained scaler on train data, to be applied to any test data before prediction
        feature_list: feature list and order that model was trained on
        '''
        self.n_inputs = n_inputs
        self.encoding_feature_sizes = encoding_feature_sizes
        self.decoding_feature_sizes = decoding_feature_sizes
        self.dropOutRate = dropOutRate
        self.encoder = None
        self.train_encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = scaler
        self.feature_list = feature_list

        # self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
        #     name="reconstruction_loss"
        # )
        # self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    # @property
    # def metrics(self):
    #     '''
    #     Metrics for learning
    #     '''
    #     return [
    #         self.total_loss_tracker,
    #         self.reconstruction_loss_tracker,
    #         self.kl_loss_tracker,
    #     ]

      def _get_training_value(self, training=None):
        if training is None:
          training = backend.learning_phase()
        if self._USE_V2_BEHAVIOR:
          if isinstance(training, int):
            training = bool(training)
          if not self.trainable:
            # When the layer is not trainable, it overrides the value passed from
            # model.
            training = False
        return training

    def call(self, data):
        '''
        Defined train_stop for the fit function of the Parent class
        '''
        z_mean, z_log_var, z = self.train_encoder(data)
        reconstruction = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        # self.metrics_tensors.append(kl_loss)
        # self.metrics_names.append("kl_loss")
        return reconstruction


    def build_encoder(self):
        '''
        Defines the encoder for training, based on the hyper parameters of the current object.
        '''

        n_input_features = self.n_inputs
        encoder_input = tf.keras.Input(shape=(n_input_features,))
        out = encoder_input
        for layer_features in self.encoding_feature_sizes[:-1]:
            out = tf.keras.layers.Dense(layer_features)(out)
            n_input_features = layer_features
            if layer_features > 100:
                out = tf.keras.layers.Dropout(rate=self.dropOutRate)(out)

        z_mean = tf.keras.layers.Dense(self.encoding_feature_sizes[-1], name="z_mean")(out)
        z_log_var = tf.keras.layers.Dense(self.encoding_feature_sizes[-1], name="z_log_var")(out)

        z = functional.Sampling()([z_mean, z_log_var])
        self.train_encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z])
        self.encoder = tf.keras.Model(encoder_input, [z])
    
    def build_decoder(self):
        '''
        Defines the decoder, based on the hyper parameters of the current object.
        Last layer gets dropout.
        '''
        n_input_features = self.encoding_feature_sizes[-1]
        decoder_input = tf.keras.Input(shape=(n_input_features,))
        out = decoder_input
        for layer_features in self.decoding_feature_sizes:
            out = tf.keras.layers.Dense(layer_features)(out)
            n_input_features = layer_features


        out = tf.keras.layers.Dense(self.n_inputs, activation="sigmoid")(out)
        self.decoder = tf.keras.Model(decoder_input, out)

    def build_autoencoder(self):
        '''
        Defines the autoencoder, based on the hyper parameters of the current object.
        '''
        if self.encoder == None:
            self.build_encoder()
        if self.decoder == None:
            self.build_decoder()
        # if self.autoencoder == None:
        #     self.autoencoder = tf.keras.models.Sequential([self.encoder, self.decoder])

    def save_model(self, hdfs_ip, path):
        '''
        Saves the model at the hdfs
        Arguments:
        hdfs_ip: The hdfs url
        path: path to save the model. Carefull, If directory does not exist it will fail
        '''
        
        # Initiate client
        client_hdfs = hdfs.InsecureClient(hdfs_ip)

        # Save encoder object as json
        with client_hdfs.write(path+'_encoder.json', overwrite=True, encoding='utf-8') \
            as writer:
            writer.write(self.encoder.to_json())

        # Save decoder object as json
        with client_hdfs.write(path+'_decoder.json', overwrite=True, encoding='utf-8') \
            as writer:
            writer.write(self.decoder.to_json())

        # Save encoder weights
        with client_hdfs.write(path+'_encoder_weights.pkl', overwrite=True) as writer:
            pickle.dump(self.encoder.get_weights(), writer)

        # Save decoder weights
        with client_hdfs.write(path+'_decoder_weights.pkl', overwrite=True) as writer:
            pickle.dump(self.decoder.get_weights(), writer)

        # Save rest of variables
        with client_hdfs.write(path+'.pkl', overwrite=True) as writer:
            pickle.dump([self.n_inputs, self.encoding_feature_sizes, \
            self.decoding_feature_sizes, self.dropOutRate, self.scaler, self.feature_list], \
            writer)

    def load_model(self, hdfs_ip, path):
        '''
        Saves the model at the hdfs
        Arguments:
        hdfs_ip: The hdfs url
        path: path to load the model from. Carefull, it must be the same path
        used with save_model function.
        '''

        # Initiate client
        client_hdfs = hdfs.InsecureClient(hdfs_ip)

        # Load all hyper parameters
        with client_hdfs.read(path+'.pkl') as reader:
            selfValues = pickle.load(reader)

        self.n_inputs = selfValues[0]
        self.encoding_feature_sizes = selfValues[1]
        self.decoding_feature_sizes = selfValues[2]
        self.dropOutRate = selfValues[3]
        self.scaler = selfValues[4]
        self.feature_list = selfValues[5]

        # Load encoder json and into a keras model
        with client_hdfs.read(path+'_encoder.json', encoding='utf-8') as reader:
            self.encoder = tf.keras.models.model_from_json(reader.read(), \
                custom_objects=functional.custom_objects)

        # Load decoder json and into a keras model
        with client_hdfs.read(path+'_decoder.json', encoding='utf-8') as reader:
            self.decoder = tf.keras.models.model_from_json(reader.read(), \
                custom_objects=functional.custom_objects)

        # Load encoder weights
        with client_hdfs.read(path+'_encoder_weights.pkl') as reader:
            self.encoder.set_weights(pickle.load(reader))

        # Load decoder weights
        with client_hdfs.read(path+'_decoder_weights.pkl') as reader:
            self.decoder.set_weights(pickle.load(reader))

        # Define autoencoder
        self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])

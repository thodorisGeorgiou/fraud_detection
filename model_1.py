import tensorflow as tf
import numpy
import pickle
import hdfs
import functional 

class model:
	'''
	Autoencoder model. Loss function depends on the user. It just defines the layers.
	Loss definition, training and predictions happen outside this class.
	'''
	def __init__(self, n_inputs, encoding_feature_sizes, decoding_feature_sizes, dropOutRate=0.1,\
		scaler=None, feature_list=None):
		'''
		Arguments:
		n_inputs: number fo input features
		encoding_feature_sizes: List of integers. Length is the number of layers and the values
		the number of nodes for the respective layer.
		decoding_feature_sizes: List of integers. Length is the number of layers and the values
		the number of nodes for the respective layer.
		dropOutRate (=0.1): Dropout rate. Defaults to 0.1
		scaler: Trained scaler on train data, to be applied to any test data before prediction
		feature_list: feature list and order that model was trained on
		'''
		self.n_inputs = n_inputs
		self.encoding_feature_sizes = encoding_feature_sizes
		self.decoding_feature_sizes = decoding_feature_sizes
		self.dropOutRate = dropOutRate
		self.encoder = None
		self.decoder = None
		self.autoencoder = None
		self.scaler = scaler
		self.feature_list = feature_list

	def build_encoder(self):
		'''
		Defines the encoder, based on the hyper parameters of the current object.
		Every layer that has more than 100 nodes gets dropout.
		'''
		self.encoder = tf.keras.models.Sequential(name='encoder')
		n_input_features = self.n_inputs

		for layer_features in self.encoding_feature_sizes[:-1]:
			self.encoder.add(layer=tf.keras.layers.Dense(units=layer_features, \
				activation=tf.keras.activations.relu, input_shape=[n_input_features]))
			n_input_features = layer_features
			if layer_features > 100:
				self.encoder.add(tf.keras.layers.Dropout(rate=self.dropOutRate))

		self.encoder.add(layer=tf.keras.layers.Dense(units=self.encoding_feature_sizes[-1], \
			activation=tf.keras.activations.relu, input_shape=[n_input_features]))

	def build_decoder(self):
		'''
		Defines the decoder, based on the hyper parameters of the current object.
		Last layer gets dropout.
		'''
		self.decoder = tf.keras.models.Sequential(name='decoder')
		n_input_features = self.encoding_feature_sizes[-1]

		for layer_features in self.decoding_feature_sizes:
			self.decoder.add(layer=tf.keras.layers.Dense(units=layer_features, \
				activation=tf.keras.activations.relu, input_shape=[n_input_features]))
			n_input_features = layer_features

		self.decoder.add(tf.keras.layers.Dropout(rate=self.dropOutRate))

		self.decoder.add(layer=tf.keras.layers.Dense(units=self.n_inputs, \
		activation=tf.keras.activations.sigmoid, input_shape=[n_input_features]))

	def build_autoencoder(self):
		'''
		Defines the autoencoder, based on the hyper parameters of the current object.
		'''
		if self.encoder == None:
			self.build_encoder()
		if self.decoder == None:
			self.build_decoder()
		if self.autoencoder == None:
			self.autoencoder = tf.keras.models.Sequential([self.encoder, self.decoder])

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

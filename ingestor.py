#Import necessary libraries
import pandas
import numpy
import hdfs
import datetime
import pickle


def save_data(indices, data, feature_list, hdfs_ip, data_path, data_name):
	# Check the validity of the input data
	if len(indices.shape) != 1:
		exit("Error: Indices array must have exactly one dimension")

	# HDFS client
	client_hdfs = hdfs.InsecureClient(hdfs_ip)

	# Get all dataset names and respective indices
	names = numpy.array([i[0] for i in indices])
	unq_names = numpy.unique(names)
	sepIndices = numpy.array([i[1] for i in indices])

	# For each dataset, contruct the csv and save it
	for name in unq_names:
		rows = numpy.where(names==name)[0]
		rowInds = sepIndices[rows]
		part_data_np = data[rows]
		part_data = pandas.DataFrame(data=part_data_np, index=rowInds, columns=feature_list)

		with client_hdfs.write(data_path+"/"+name+"/"+data_name+".csv", overwrite=True, \
			encoding='utf-8') as writer:
			part_data.to_csv(writer)

class ingestor:
	'''A class that keeps a connection to the hdfs and track of know features.
	Is responsible for ingesting new data'''
	def __init__(self, hdfs_ip, data_path="/user/anomaly", test_ratio=0.2):
		'''Arguments

		hdfs_ip: is a string with the IP and port of the hdfs, e.g. http://localhost:9870
		data_path(="/user/anomaly"): The path in the hdfs for the datasets
		test_ratio(=0.2): float, the proportion of the data to become test set.
		'''
		self.data_path = data_path
		self.hdfs_ip = hdfs_ip
		self.test_ratio = test_ratio
		#Start connection to hdfs
		self.client_hdfs = hdfs.InsecureClient(self.hdfs_ip)
		#Check if there are data on the server. If not this is the first job
		self.existing_data = self.client_hdfs.list(self.data_path)
		#Keep a list of known features
		if len(self.existing_data) != 0:
			with self.client_hdfs.read(self.data_path+'/feature_list.pkl') as reader:
				self.known_features = pickle.load(reader)
		else:
			self.known_features = []

	def splitToSets(self, data):
		'''Splits a numpy array to train and test sets. The default proportions is 20% to 
		test set
		Arguments:
		data: The dataset to be split. Expected Pandas dataframe
		'''
		#Negative samples
		allNegative = numpy.where(data["Class"].values==0)[0]
		numpy.random.shuffle(allNegative)

		#Positive samples
		allPositive = numpy.where(data["Class"].values==1)[0]
		numpy.random.shuffle(allPositive)

		#Cutoff index for splitting to train and test sets
		negative_cutoff = int(self.test_ratio*allNegative.shape[0])
		positive_cutoff = int(self.test_ratio*allPositive.shape[0])

		#Create the splits
		test_negative = allNegative[:negative_cutoff]
		train_negative = allNegative[negative_cutoff:]
		test_positive = allPositive[:positive_cutoff]
		train_positive = allPositive[positive_cutoff:]
		train_data = data.iloc[numpy.concatenate([train_positive, train_negative])]
		test_data = data.iloc[numpy.concatenate([test_positive, test_negative])]

		#Correct original indices
		train_data = train_data.reset_index()
		train_data = train_data.drop('index', axis='columns')
		test_data = test_data.reset_index()
		test_data = test_data.drop('index', axis='columns')

		return train_data, test_data

	def mapFeatures(self, new_features):
		'''Creates index mapping of the new feature columns with respect to the known ones.
		Assumes that if two features have the same name, they are the same
		'''
		#Initial objects and needed constants
		n_known_features = len(self.known_features)
		n_new_features = 0
		indices = []
		unseen_features = []
		
		#Check the index order of the new dataset with respect to the old dataset and identify 
		# the features that have not been seen before
		for feature in new_features:
			try:
				indices.append(self.known_features.index(feature))
			except ValueError:
				indices.append(n_known_features+n_new_features)
				n_new_features += 1
				unseen_features.append(feature)
		
		#Find if there are previously known features that are missing from this dataset
		missing = numpy.sort(list(set(list(range(n_known_features))) - set(indices)))
		
		#Account for them in the index array
		for ni, oi in enumerate(indices):
			indices[ni] = oi - numpy.searchsorted(missing, oi)
		
		return indices, unseen_features

	def expand_time(self, data):
		'''Creates three new columns in data, one expressing time in radians, 
		one the cosine and one the sinus of the time.
		Arguments:
		data: pandas dataframe that has a column named "Time"
		'''
		#Mapping from seconds to radians 
		correction = 2*numpy.pi/(3600*24)
		#Calculate radians 
		data["rad_time"] = data["Time"]*correction
		#Calculate cosine 
		data["cos_time"] = numpy.cos(data["rad_time"])
		#Calculate sinus 
		data["sin_time"] = numpy.sin(data["rad_time"])
		return data

	def ingest(self, data_file):
		'''Ingest new files.
		Arguments:
		data_file: Gets a string path to a cv file with the new data to ingest.
		'''
		#Read input csv. If not possible exit with error.
		try:
			new_data = pandas.read_csv(data_file)
		except pandas.errors.ParserError:
			exit("Error: Input file can not be read as a csv file")

		#Check if the dataset is labeled. If not ingestion is not done
		if "Class" not in new_data.columns:
			exit("Error: There is no column called -Class-. The dataset is either not labeled \
				or the class column has a different name. Apadpt the incoming data and\
				 try again.")
		
		#Add new feature columns, time transformations
		if "Time" in new_data.columns:
			new_data = self.expand_time(new_data)

		#Change the order of the features according to the known features order
		new_features = new_data.keys().tolist()
		#Get index order of features in new dataset and check for potential new features
		indices, unseen_features = self.mapFeatures(new_features)
		if len(unseen_features) > 0:
			self.known_features += unseen_features
			#Update known features list
			with self.client_hdfs.write(self.data_path+'/feature_list.pkl', overwrite=True) as writer:
				pickle.dump(self.known_features, writer)

		#Reshuffle columns
		new_data = new_data[numpy.array(new_features)[indices]]

		#Timestamp identifier of data chunk
		new_data_name = self.data_path+"/"+str(datetime.datetime.utcnow().timestamp())
		self.client_hdfs.makedirs(new_data_name)

		#Split dataset to train and test sets
		train_set, test_set = self.splitToSets(new_data)

		#Write train and test sets to hdfs
		with self.client_hdfs.write(new_data_name+'/train_set.csv', encoding='utf-8') as writer:
			train_set.to_csv(writer)

		with self.client_hdfs.write(new_data_name+'/test_set.csv', encoding='utf-8') as writer:
			test_set.to_csv(writer)

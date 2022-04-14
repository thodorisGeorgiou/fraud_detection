#Import necessary libraries
import pandas
import numpy
import hdfs
import datetime
import pickle


class data_loader:
	'''A class that keeps a connection to the hdfs and track of know features.
	Is responsible for ingesting new data'''
	def __init__(self, hdfs_ip, data_path="/user/anomaly"):
		'''Arguments

		hdfs_ip: is a string with the IP and port of the hdfs, e.g. http://localhost:9870
		data_path(="/user/anomaly"): The path in the hdfs for the datasets
		test_ratio(=0.2): float, the proportion of the data to become test set.
		'''
		self.data_path = data_path
		self.hdfs_ip = hdfs_ip
		#Start connection to hdfs
		self.client_hdfs = hdfs.InsecureClient(self.hdfs_ip)
		#Check if there are data on the server. If not this is the first job
		self.existing_data = self.client_hdfs.list(self.data_path)


	def get_train_data(self, feature_list=[]):
		'''Get all train data.
		Arguments:
		feature_list: Bring back data points that have all the features in the feature list.
		Can be used for training different models in different subsets of features. Default is
		empty list in which case the returned data is all the data points with the subset of 
		features that exist for all data points.
		'''
		if "feature_list.pkl" not in self.existing_data:
			exit("The HDFS is corrupted or no data have been inserted. Please make sure everything\
				is in order.")

		#Leave only dataset directories on the list
		# self.existing_data.remove('feature_list.pkl')
		if len(feature_list) == 0:
			with self.client_hdfs.read(self.data_path+'/feature_list.pkl') as reader:
				feature_list = pickle.load(reader)


		train_data = []
		directories = list(self.existing_data)
		directories.remove('feature_list.pkl')
		#Go through all train set csv files and keep the appropriate part
		for directory in directories:
			with self.client_hdfs.read(self.data_path+'/'+directory+"/train_set.csv", encoding='utf-8') as reader:
				part_data = pandas.read_csv(reader, index_col=0)
				try:
					train_data.append(part_data[feature_list])
				except KeyError:
					continue

		#Concatenate all dataframes
		train_data = pandas.concat(train_data, keys=directories)

		return train_data

	def get_test_data(self, feature_list=[]):
		'''Get all test data.
		Arguments:
		feature_list: Bring back data points that have all the features in the feature list.
		Can be used for testing different models in different subsets of features. Default is
		empty list in which case the returned data is all the data points with the subset of 
		features that exist for all data points.
		'''
		if "feature_list.pkl" not in self.existing_data:
			exit("The HDFS is corrupted or no data have been inserted. Please make sure everything\
				is in order.")

		#Leave only dataset directories on the list
		# self.existing_data.remove('feature_list.pkl')
		if len(feature_list) == 0:
			with self.client_hdfs.read(self.data_path+'/feature_list.pkl') as reader:
				feature_list = pickle.load(reader)


		test_data = []
		directories = list(self.existing_data)
		directories.remove('feature_list.pkl')
		#Go through all test set csv files and keep the appropriate part
		for directory in directories:
			with self.client_hdfs.read(self.data_path+'/'+directory+"/test_set.csv", encoding='utf-8') as reader:
				part_data = pandas.read_csv(reader, index_col=0)
				try:
					test_data.append(part_data[feature_list])
				except KeyError as e:
					continue

		# Check whether there are data with sufficient columns
		if len(test_data) == 0:
			return None

		#Concatenate all dataframes
		test_data = pandas.concat(test_data, keys=directories)

		return test_data

	def get_data(self, name, feature_list=[]):
		'''Get all test data.
		Arguments:
		name: the name of the dataset, e.g. "train_set", "test_set", "test_model_reconstructed"
		feature_list: Bring back data points that have all the features in the feature list.
		Can be used for testing different models in different subsets of features. Default is
		empty list in which case the returned data is all the data points with the subset of 
		features that exist for all data points.
		'''
		if "feature_list.pkl" not in self.existing_data:
			exit("The HDFS is corrupted or no data have been inserted. Please make sure everything\
				is in order.")

		#Leave only dataset directories on the list
		# self.existing_data.remove('feature_list.pkl')
		if len(feature_list) == 0:
			with self.client_hdfs.read(self.data_path+'/feature_list.pkl') as reader:
				feature_list = pickle.load(reader)


		data = []
		directories = list(self.existing_data)
		directories.remove('feature_list.pkl')
		#Go through all test set csv files and keep the appropriate part
		for directory in directories:
			with self.client_hdfs.read(self.data_path+'/'+directory+"/"+name+".csv", encoding='utf-8') as reader:
				part_data = pandas.read_csv(reader, index_col=0)
				try:
					data.append(part_data[feature_list])
				except KeyError as e:
					continue

		# Check whether there are data with sufficient columns
		if len(data) == 0:
			return None

		#Concatenate all dataframes
		data = pandas.concat(data, keys=directories)

		return data


import sys
import numpy
import pandas
import tensorflow as tf
import sklearn.metrics as metrics

import data_loader
import ingestor
import model_1

model_name = sys.argv[1]
# model_name = "model1"

# HDFS Url, where all our data and models are
hdfs_ip = sys.argv[2]
# hdfs_ip = "http://localhost:9870"

# Define and load trained model
model = model_1.model(0, [150, 75, 4], [75, 150])
model.load_model(hdfs_ip, "/user/models/"+model_name)

#Load test data (could also work with a listener for live application)
dl = data_loader.data_loader(hdfs_ip)
test_set = dl.get_test_data(model.feature_list+["Class"])
indices = test_set.index

#Extract ground truth
ground_truth = test_set["Class"].values
test_set = test_set.drop("Class", axis='columns')

#Rescale data according to the expectations of the model
test_set = test_set.values
test_set = model.scaler.transform(test_set)

#Calculate reconstruction, error and performance measures
test_reconstructed = model.autoencoder.predict(test_set)
test_loss = numpy.mean((test_reconstructed-test_set)**2, axis=1)

#False positive rate, true positive rate and respective thresholds
fpr, tpr, threshold = metrics.roc_curve(ground_truth, test_loss)

#Area under the curve
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

# Add loss to reconstructed array, as last column
test_model_data = numpy.concatenate([test_reconstructed, numpy.expand_dims(test_loss, -1)], \
	axis=-1)

# Save produced data
ingestor.save_data(indices, test_model_data, model.feature_list+["reconstruction_mse"], \
	hdfs_ip, "/user/anomaly/", "train_reconstructed_"+model_name)



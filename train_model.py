import sys
import numpy
import pandas
import tensorflow as tf
import data_loader
import model_1
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

model_name = sys.argv[1]
# hdfs_ip = sys.argv[2]
# model_name = "variational_1"
hdfs_ip = "http://localhost:9870"

#Load train set
dl = data_loader.data_loader(hdfs_ip)
train_set = dl.get_train_data().drop(["Time", "rad_time"], axis='columns')


#Shuffle data for splitting (classes are separable by index)
#If we have already shuffled in the past, use the same shuffling 
#for better comparison of models
shuffle_indices = numpy.arange(train_set.values.shape[0])
try:
	shuffle_indices = numpy.load("shuffled_indices.npy")
except FileNotFoundError:
	numpy.random.shuffle(shuffle_indices)
	numpy.save("shuffled_indices.npy", shuffle_indices)

#Separate class column
ground_truth = train_set["Class"]
train_set = train_set.drop("Class", axis='columns')
feature_list = train_set.keys().tolist()

#Make training and validation splits
train_set = train_set.values
cutoff = int(train_set.shape[0]*0.2)
val_indices, train_indices = numpy.split(shuffle_indices, [cutoff])

val_set = train_set[val_indices]
train_set = train_set[train_indices]

val_ground_truth = ground_truth[val_indices]
train_ground_truth = ground_truth[train_indices]

#Scale data to be in the range of [0,1]
scaler = MinMaxScaler()
data_scaled = scaler.fit(train_set)
train_set = data_scaled.transform(train_set)
val_set = data_scaled.transform(val_set)

#Build model
model = model_1.model(train_set.shape[1], [150, 75, 4], [75, 150], dropOutRate=0.1,\
	scaler=scaler, feature_list=feature_list)
model.build_autoencoder()
model.autoencoder.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(), \
	metrics=[tf.keras.metrics.mean_squared_error])


#Set criteria to stop training
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, \
	restore_best_weights=True)

#Train model
history = model.autoencoder.fit(x=train_set, y=train_set, epochs=100, verbose=1, \
	validation_data=(val_set, val_set), callbacks=[es])

#Evaluate model
train_reconstructed = model.autoencoder.predict(train_set)
val_reconstructed = model.autoencoder.predict(val_set)

#Mean square error
train_loss = numpy.mean((train_reconstructed-train_set)**2, axis=1)
val_loss = numpy.mean((val_reconstructed-val_set)**2, axis=1)

#False positive rate, true positive rate and respective thresholds
fpr, tpr, threshold = metrics.roc_curve(val_ground_truth.values, val_loss)

#Area under the curve
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

# Save model to hdfs
model.save_model(hdfs_ip, "/user/models/"+model_name)

Fraud detection exercise - Python 3.6.9

Prerequisites (tested version):

	Libraries:
			Numpy			(1.19.5)
			Pandas			(0.24.2)
			hdfs			(2.6.0)
			sklearn			(0.19.2)
			tensorflow		(1.13.2)

	Setup:
			Working HDFS with two directories:
				/user/anomaly
				/user/models

Includes:

	Scripts:

		ingest.py:

			Takes two arguments:	Path to  csv file to be ingested
									hdfs url

			example usage:

					python3 ingest.py /example/path/creditcard.csv http://localhost:9870



		train_model.py:

			Takes two arguments:	model name. A unique name to give to the model.
									If the name already exists, it will be replaced

									hdfs url. Url of the HDFS that the model will be saved

									Besides training, it also stores on the HDFS
									the reconstructed entries together with the respective MSE


		fraud_detection.py:

			Takes two arguments:	model name. The name of the model to use

									hdfs url. Url of the HDFS from which data and model will be loaded.

									Besides getting the validation loss, it also stored on the HDFS
									the reconstructed entries together with the respective MSE

	Important Classes:

		ingestor.py:

			Responsible for ingesting the data to the HDFS and any preprocessing step required before storing


		data_loader.py:

			Loads the requested data from the HDFS

		model_1.py:

			Definition of the autoencoder

			IMPORTNANT: This class can be changed if a completely different architecture is needed.
			As long as the "save_model" and "load_model" methods stay intact, any new or old model can be loaded, even if the rest of the class has changed.
			Convenient if multiple models are used in the final pipeline

		variational_autoencoder.py:

			Definition of the variational autoencoder

			IMPORTNANT: This class is adapted from model_1. After training and saving a model,
			the models can be loaded from either the variational autoencoder class or the basic model class.

	Other:

		functional.py

			Contains small, generic functions and classes

			IMPORTANT: If you create a new class for a specific layer of your model, this class
			has to be added to functional.custom_objects, otherwise keras won't be able to load
			saved models that utilize them. 


Short description:

Every incoming csv file, is getting its own unique identifier, the timestamp of the ingestion time.
We are keeping a global list of all know features and their order. For a new csv we re-order the features according to the aforementioned order and add the new to the global list. Three new rows are created related to time. We map time from [0, 24) hours to [0, 2π). Then the time entry is transformed to radians and the cosine and sinus are computed. Finally, the csv is split to train and test data and added to the HDFS in its own directory.

Data are loaded, with a required list of features and parts of data (e.g. train, test, train_reconstructed etc.). All csv files that include the requested list of features are concatenated and returned. This data loading can be done for training, validating and product deployment aka in real world

The training script is requesting all training data with a specific feature list.
It is split to 80% train and 20% validation data. Transforms them using a Min - Max normalization (with respect to the train data). The autoencoder is trained using the train set for training and validation set for early stopping. The resulted model is saved on the HDFS for further use. False positive rate, true positive rate and ROC area under the curve are calculated and printed. (The pipeline is supposed to be completely unsupervised so the last step can be deleted for deployment)

The fraud detection script can loads train or test set and passes it through a saved model. The reconstruction, and reconstruction errors are concatenated into a single matrix and saved on the HDFS. False positive rate, true positive rate and ROC area under the curve are calculated and printed.
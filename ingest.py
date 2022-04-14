import sys
import numpy
import pandas
import ingestor

#Get input argument
input_file = sys.argv[1]
# input_file = "creditcard.csv"

hdfs_ip = sys.argv[2]
# hdfs_ip = "http://localhost:9870"
#Initiate instance
ing = ingestor.ingestor(hdfs_ip)

#Ingest
ing.ingest(input_file)
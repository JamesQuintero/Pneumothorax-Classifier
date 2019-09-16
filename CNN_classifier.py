from DICOM_reader import DICOMReader
from DataHandler import DataHandler


"""

Handles CNN training, validation, and testing

"""
class CNNClassifier:

	dicom_reader = None
	data_handler = None

	def __init__(self):
		self.dicom_reader = DICOMReader()
		self.data_handler = DataHandler()


	#returns CNN
	def create_CNN(self):
		pass

	#trains CNN
	def train(self):

		#labels are dataframe where keys are column names, and values are rows of that column
		labels = self.data_handler.read_train_labels()

		#converts DataFrame to 2D list
		labels = labels.values.tolist()

		print("Num labels: "+str(len(labels)))


		train_ratio = 0.5
		validation_ratio = 0.2
		train_labels, validation_labels, test_labels = self.data_handler.split_data(labels, train_ratio, validation_ratio)

		# for label in labels:
		# 	print(label)




if __name__=="__main__":
	CNN_classifier = CNNClassifier()

	CNN_classifier.train()
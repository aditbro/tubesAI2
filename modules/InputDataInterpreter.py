import csv

class InputDataInterpreter():
	def __init__(self, filename = ""):
		self.filename = filename
		self.data = []
		self.target = []
		self.processInputFile()

	def processInputFile(self):
		input_data = self.getInputFileContent()
		self.makeDatasetList(input_data)

	def getInputFileContent(self):
		data_content = []
		
		with open(self.filename, newline='') as csvfile:
			file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
			
			for row in file_content:
				content_row = []
				for data in row:
					content_row.append(data)
				data_content.append(content_row)

		return data_content

	def makeDatasetList(self, input_data):
		for row in input_data:
			self.target.append(row[-1])
			self.data.append(row[0:-1])

		self.patchUnknownData()

	def patchUnknownData(self):
		column_patch_method = ["median", "modus", "modus", "mean", \
		"mean", "modus", "modus", "median", \
		"modus", "median", "modus", "modus", \
		"modus", "median"]

		column_patch_values = self.getColumnPatchVal(column_patch_method)

		for i in range(0, len(self.data)):
			for j in range(0,len(self.data[0])):
				if self.data[i][j] == '?':
					self.data[i][j] = column_patch_values[j]


inp = InputDataInterpreter(filename="../data/tubes2_HeartDisease_train.csv")
print(inp.getInputFileContent())
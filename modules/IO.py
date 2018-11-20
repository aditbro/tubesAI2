import csv
from sklearn.impute import SimpleImputer

class InputDataInterpreter():
	def __init__(self, filename = ""):
		self.filename = filename
		self.data = []
		self.target = []
		self.processInputFile()

	def reduceUnknownData(self):
		for row in self.data:
			if(self.countUnknownAttr(row) > 13):
				self.data.remove(row)

	def countUnknownAttr(self, row):
		count_unknown = 0
		for data in row:
			if data == '?':
				count_unknown += 1
		
		return count_unknown

	def processInputFile(self):
		input_data = self.getInputFileContent()

		self.makeDatasetList(input_data)
		
		for i in range(len(self.data)):
			self.target[i] = int(self.target[i])
			for j in range(len(self.data[0])):
				self.data[i][j] = float(self.data[i][j])	

	def getInputFileContent(self):
		data_content = []
		
		with open(self.filename, newline='') as csvfile:
			file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
			
			for row in file_content:
				content_row = []
				for data in row:
					content_row.append(data)
				data_content.append(content_row)

		return data_content[1:]

	def makeDatasetList(self, input_data):
		for row in input_data:
			self.target.append(row[0].split(',')[-1])
			self.data.append(row[0].split(',')[0:13])
		
		self.reduceUnknownData()
		
		self.patchUnknownData()

	def patchUnknownData(self):
		column_patch_method = ["median", "modus", "modus", "mean", \
		"mean", "modus", "modus", "mean", \
		"modus", "mean", "modus", "modus", "modus"]
		
		column_patch_values = self.getColumnPatchVal(column_patch_method)

		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				if self.data[i][j] == '?' or self.data[i][j] == '':
					self.data[i][j] = column_patch_values[j]


	def getColumnPatchVal(self, patch_method):
		patch_val = []

		for i in range(len(patch_method)):
			if patch_method[i] == 'modus':
				patch_val.append(self.getDataModus(i))
			elif patch_method[i] == 'median':
				patch_val.append(self.getDataMedian(i))
			elif patch_method[i] == 'mean':
				patch_val.append(self.getDataMean(i))

		return patch_val

	def getDataModus(self, j):
		data_dict = {}

		for i in range(len(self.data)):
			if self.data[i][j] == '?':
				continue
			if str(self.data[i][j]) in data_dict:
				data_dict[str(self.data[i][j])] += 1
			else :
				data_dict[str(self.data[i][j])] = 0

		max_key = ''
		max_val = -1
		for key, val in data_dict.items():
			if val > max_val:
				max_key = key
				max_val = val

		return str(max_val)

	def getDataMedian(self, j):
		column_list = []

		for i in range(len(self.data)):
			if self.data[i][j] == '?':
				continue
			column_list.append(self.data[i][j])

		column_list.sort()
		median_idx = (len(column_list)//2) + 1

		return str(column_list[median_idx])

	def __is_int__(self, input):
		try:
			a = int(input)
			return True
		except :
			return False

	def getDataMean(self, j):
		column_sum = 0

		for i in range(len(self.data)):
			if self.__is_int__(self.data[i][j]):
				column_sum += int(self.data[i][j])

		return str(column_sum / len(self.data))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class TestDataInterpreter():
	def __init__(self, filename = ""):
		self.filename = filename
		self.data = []
		self.processInputFile()

	def processInputFile(self):
		input_data = self.getInputFileContent()
		self.makeDatasetList(input_data)
		
		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				self.data[i][j] = float(self.data[i][j])	

	def getInputFileContent(self):
		data_content = []
		
		with open(self.filename, newline='') as csvfile:
			file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
			
			for row in file_content:
				content_row = []
				for data in row:
					content_row.append(data)
				data_content.append(content_row)

		return data_content[1:]

	def makeDatasetList(self, input_data):
		for row in input_data:
			self.data.append(row[0].split(','))

		self.patchUnknownData()

	def patchUnknownData(self):
		column_patch_method = ["median", "modus", "modus", "mean", \
		"mean", "modus", "modus", "median", \
		"modus", "median", "modus", "modus", \
		"modus"]

		column_patch_values = self.getColumnPatchVal(column_patch_method)

		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				if self.data[i][j] == '?':
					self.data[i][j] = column_patch_values[j]


	def getColumnPatchVal(self, patch_method):
		patch_val = []

		for i in range(len(patch_method)):
			if patch_method[i] == 'modus':
				patch_val.append(self.getDataModus(i))
			elif patch_method[i] == 'median':
				patch_val.append(self.getDataMedian(i))
			elif patch_method[i] == 'mean':
				patch_val.append(self.getDataMean(i))

		return patch_val

	def getDataModus(self, j):
		data_dict = {}

		for i in range(len(self.data)):
			if str(self.data[i][j]) in data_dict:
				data_dict[str(self.data[i][j])] += 1
			else :
				data_dict[str(self.data[i][j])] = 0

		max_key = ''
		max_val = -1
		for key, val in data_dict.items():
			if val > max_val:
				max_key = key
				max_val = val

		return str(max_val)

	def getDataMedian(self, j):
		column_list = []

		for i in range(len(self.data)):
			column_list.append(self.data[i][j])

		column_list.sort()
		median_idx = (len(column_list)//2) + 1

		return str(column_list[median_idx])

	def __is_int__(self, input):
		try:
			a = int(input)
			return True
		except :
			return False

	def getDataMean(self, j):
		column_sum = 0

		for i in range(len(self.data)):
			if self.__is_int__(self.data[i][j]):
				column_sum += int(self.data[i][j])

		return str(column_sum / len(self.data)) 
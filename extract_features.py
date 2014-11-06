

DataSet = []
Y_data = []

TrainSet = []
TestSet = []
Y_train = []
Y_test = []

NumMissing = []
N = 3500
P = 1897

NumMissing = [0 for i in range(P)]
NumClass1 = 0
NumClass2 = 0 
NumDataNoMissing = 0

#2 class with labels 1 and 2
f = open("ML-contest-training/contest_train.csv","r")
for line in f:
	miss = 0
	vector = line.split(",")
	for i in range(P):
		if vector[i] == "NaN":
			NumMissing[i] += 1
			miss += 1
		else:
			vector[i] = float(vector[i])
	i += 1
	DataSet.append(vector[0:P])
	vector[i] = int(vector[i])
	
	Y_data.append(vector[i])
	if miss == 0:
		NumDataNoMissing += 1 
		if vector[i]==1:
			NumClass1 += 1
		else:
			NumClass2 += 1

print NumDataNoMissing
AvgMissing = 0
MaxMissing = 0
for val in NumMissing:
	AvgMissing += val 
	if val > MaxMissing:
		MaxMissing = val
AvgMissing = float(AvgMissing)/P

print "N :",N,"P :",P
print "Class 1 :",NumClass1,"Class 2 :",NumClass2
print "Missing Features Max and Avg :", MaxMissing, AvgMissing

# should be ok, since max missing is only 10
def impute_mean(DataSet):
	SampleMean = [0.0 for i in range(P)]
	for point in DataSet:
		for f in range(P):
			if point[f] != "NaN":
				SampleMean[f] += point[f]

	SampleMean = [ SM/P for SM in SampleMean]
	for point in DataSet:
		for f in range(P):
			if point[f] == "NaN":
				point[f] = SampleMean[f] 

	return DataSet

TestSetNum = 7	#ratio of DataSet : TestSet
impute_mean(DataSet)
for i in range(N):
	if i%TestSetNum == 0:
		TestSet.append(DataSet[i])
		Y_test.append(Y_data[i])
	else :
		TrainSet.append(DataSet[i])
		Y_train.append(Y_data[i])

print len(TrainSet),len(TestSet)
# data has been split into TrainSet and TestSet


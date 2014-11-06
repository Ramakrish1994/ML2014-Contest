#function which computes statistics associated with linear classification for binary classes
#tp - true positives
#tn - true negatives
#fn - false negatives
#fp - false positives
def collect_stat(I,O):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in range(len(I)):
		if I[i]==2:
			if O[i]==2:
				tp += 1
			else:
				fp += 1
		else:
		
			if O[i]==2:
				fn += 1
			else:
				tn += 1

	
	accuracy = (float)(tp + tn)/(tp + tn + fn + fp)
	print "Accuracy :"
	print accuracy	
	print "A -"
	precisionA = (float)(tp)/(tp+fp)
#print precisionA
	recallA = (float)(tp)/(tp+fn)
#	print recallA
	fmeasureA = 2*precisionA*recallA/(precisionA+recallA)
#	print fmeasureA	
	print "B -"
	precisionB = (float)(tn)/(tn+fn)
#	print precisionB
	recallB = (float)(tn)/(tn+fp)
#	print recallB
	fmeasureB = 2*precisionB*recallB/(precisionB+recallB)
#	print fmeasureB
	hmf = 2*fmeasureB*fmeasureB/(fmeasureB + fmeasureA)
	print "HMF -"+hmf	


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

import numpy as np
from sklearn.lda import LDA

clf = LDA()
clf.fit(np.array(TrainSet),np.array(Y_train))

output = clf.predict(TestSet)
collect_stat(Y_test, output)	



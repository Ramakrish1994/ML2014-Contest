The training set consists of 3500 data points with 1897 real-valued features. There are some missing values in the dataset and they are represented as NaN's. The last column in the training set is the class label. It is a two-class problem with the labels being 1 and 2.

The performance measure that we would be using will be the harmonic mean of the F1-scores of the 2 classes, i.e., if f1 is the F1-score of class 1, and f2 the score of class 2, then performance measure = 2 / ((1/f1) + (1/f2)).

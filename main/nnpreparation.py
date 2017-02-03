'''
Obucavanje neuronske mreze prema ugledu na skripte sa sajta

handwritten-digit-recognition-using-opencv-skleaerner-and-python
'''

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

dataset = datasets.fetch_mldata('MNIST Original')
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')
list_hog_fd = []
print("Prepearing features")
for feature in features:
    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
print("HOG features prepared successfully")
clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)
print("Clasifier made and ready")
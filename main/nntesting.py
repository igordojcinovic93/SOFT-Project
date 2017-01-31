'''
Modul namenjen za testiranje obucene neuronske mreze

Igor Dojcinovic
'''

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import nnprep as nn
plt.interactive(False)


print matplotlib.get_backend()

imgNum = 2
nn.prep()
test_data = nn.get_test_data()
img = test_data[imgNum]
print img.shape
img = img.reshape(28,28)
plt.imshow(img, cmap="Greys")
plt.show()
model = nn.get_model()
nn.train_nn()
t = model.predict(test_data, verbose=1)
print t[imgNum]

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xticks(x)
width = 1/1.5
plt.bar(x,t[imgNum], color="blue")
plt.show()
rez = t.argmax(axis=1)
print rez[imgNum]
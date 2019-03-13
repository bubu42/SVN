import numpy as np
from sklearn.svm import SVC  # import de la classe SVC pour SVM
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt  # the library for plotting

# 2 SVM with Scikit-learn
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # we create 4 examples
y = np.array([-1, -1, 1, 1])

classif = SVC()  # we create a SVM with default parameters
classif.fit(X, y)  # we learn the model according to given data
res = classif.predict([[-0.8, -1]])  # prediction on a new sample
print(res)

# we create a mesh to plot in
h = .02  # grid step
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# the grid is created, the intersections are in xx and yy
mysvc = SVC(kernel="rbf", C=2.0)
mysvc.fit(X, y)
Z2d = mysvc.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d = Z2d.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
# We plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

# DATASET GENERATION


X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2,
                           random_state=2, n_clusters_per_class=1)
plt.figure()
plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
plt.savefig("plt_classification")


# 3.2
X, y = make_moons(noise=0.1, random_state=1, n_samples=40)

plt.figure()
plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.show()
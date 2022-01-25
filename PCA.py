from sklearn import datasets
iris=datasets.load_iris()
print(iris)
#bagi atribu dan labels
atribut=iris.data
print(atribut)#panjang setiap komponen iris
label=iris.target
print(label)#spesices
#bagi dataset menjadi train dan set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(atribut,label,test_size=0.2,random_state=1)
#untuk trainnya pakai model desicision tree yg biasanya digunakan untuk klarifikasi
from sklearn import tree
descision_tree=tree.DecisionTreeClassifier()
model_pertama=descision_tree.fit(X_train,Y_train)
model_pertama=model_pertama.score(X_test,Y_test)
print(model_pertama)
#0.96
#kemudian kita gunakan pca, biasanya digunakan jika data set terlalu banyak
from sklearn.decomposition import PCA
#membuat PCA dengan 4 Principal component
pca=PCA(n_components=4)
#mengaplikasikan PCA pada data set
pca_atribut=pca.fit(X_train)
pca_atribut=pca.transform(X_train)
pca_atribut=pca.explained_variance_ratio_
print(pca_atribut)
#ambil dua yg terbaik->dimensionality reduction.
pca=PCA(n_components=2)
X_train_pca=pca.fit(X_train)
X_train_pca=X_train_pca.transform(X_train)
X_test_pca=pca.fit(X_test)
X_test_pca=X_test_pca.transform(X_test)


#uji akurasi classifier dengan pca
model2=descision_tree
model2=model2.fit(X_train_pca,Y_train)
model2=model2.score(X_test_pca,Y_test)
print(model2)
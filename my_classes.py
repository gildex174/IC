import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.spatial.distance import pdist


from sortedness import sortedness, global_pwsortedness

from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits


from sklearn.preprocessing import StandardScaler

import umap
from sklearn.manifold  import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,Isomap
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.model_selection import train_test_split,  KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix

#Logistic Regression:
from sklearn.linear_model import LogisticRegression
#Support Vector Machines (SVM):
from sklearn.svm import SVC
#Random Forest:
from sklearn.ensemble import RandomForestClassifier
#K-Nearest Neighbors (KNN):
from sklearn.neighbors import KNeighborsClassifier
#Decision Tree:
from sklearn.tree import DecisionTreeClassifier
#Gradient Boosting:
from sklearn.ensemble import GradientBoostingClassifier
#Gaussian Naive Bayes:
from sklearn.naive_bayes import GaussianNB
#AdaBoost:
from sklearn.ensemble import AdaBoostClassifier
#MLP
from sklearn.neural_network import MLPClassifier

class MyActLearning():

    def __init__(self, n_iterations=5, k_samples=10):
        self.n_iterations = n_iterations
        self.k_samples = k_samples
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    def fit(self, X_pool, y_pool):
        test_size_aux = X_pool.shape[0] - (0+1)*10
        X_train, X_val, y_train, y_val = train_test_split(X_pool, y_pool, test_size=test_size_aux, random_state=42)

        for i in range(self.n_iterations):
            model = self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_val)

            # Obter as probabilidades das classes para o conjunto de teste
            probabilities = model.predict_proba(X_val)

            # Calcular a diferença entre as probabilidades das classes
            differences = np.abs(probabilities[:, 0] - probabilities[:, 1])

            # Ordenar os exemplos pelo valor absoluto da diferença (do menor para o maior)
            sorted_indices = np.argsort(differences)

            # Exibir os 10 exemplos mais incertos
            most_uncertain_indices = sorted_indices[:self.k_samples]

            X_train = np.concatenate((X_train, X_val[most_uncertain_indices]))
            y_train = np.concatenate((y_train, y_val[most_uncertain_indices]))

            # print(f'Shape do X_train: {X_train.shape[0]}')

            X_val = np.delete(X_val, most_uncertain_indices, axis=0)
            y_val = np.delete(y_val, most_uncertain_indices)
    
    def get_params(self, deep=True):
        # Retorna os parâmetros do estimador em um dicionário
        return {"n_iterations": self.n_iterations, "k_samples": self.k_samples}
    
    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred
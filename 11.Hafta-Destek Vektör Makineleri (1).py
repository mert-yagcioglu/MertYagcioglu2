#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X = data.data
y = data.target



#Burada Normalizasyon yapiyoruz
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


#Fold degerlerini burada ekleyip azaltabiliriz
n_folds_list = [3]




#Kernel fonksiyonları
kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']




for n_folds in n_folds_list:
    print(f"\nNumber of Folds: {n_folds}\n{'='*40}")

    for k in range(1, 2):
        for metric in ['minkowski', 'manhattan']:
            for kernel in kernel_list:
                knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')

                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                y_prob_all = []
                y_true_all = []

                for train_index, test_index in kf.split(X_normalized, y):
                    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    knn_classifier.fit(X_train, y_train)

                    y_prob = knn_classifier.predict_proba(X_test)[:, 1]
                    y_prob_all.extend(y_prob)
                    y_true_all.extend(y_test)

#ROC eğrisi
                fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)                          
                auc_value = auc(fpr, tpr)

                plt.plot(fpr, tpr, label=f'Folds={n_folds}, k={k}, metric={metric}, kernel={kernel} (AUC = {auc_value:.2f})')

                
                
                
# Accuracy,recall,precision,f1 score,confusion matrix degerlerini burada hesapliyoruz.
                y_pred = (np.array(y_prob_all) > 0.5).astype(int)
                accuracy = accuracy_score(y_true_all, y_pred)
                recall = recall_score(y_true_all, y_pred)
                precision = precision_score(y_true_all, y_pred)
                f1 = f1_score(y_true_all, y_pred)
                cm = confusion_matrix(y_true_all, y_pred)

                
                
                print(f"K-NN - Folds={n_folds}, k={k}, metric={metric}, kernel={kernel}")
                print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
                print("Confusion Matrix:")
                print(cm)
                print(f"AUC: {auc_value:.4f}\n")

                
                
plt.plot([0, 1], [0, 1], '--', color='gray', label='Rastgele')
plt.xlabel('False Positive Orani')
plt.ylabel('True Positive Orani')
plt.title('Farkli Parametreler icin ROC CURVE')
plt.legend()
plt.show()







#KAYNAKCA:
#https://scikit-learn.org/stable/modules/svm.html
#https://tr.wikipedia.org/wiki/Destek_vekt%C3%B6r_makinesi
#https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
#https://www.geeksforgeeks.org/support-vector-machine-algorithm/
#https://www.bilimma.com/destek-vektor-makineleri-support-vectors-machines-svms/
#https://iksadyayinevi.com/wp-content/uploads/2020/12/MAKINE-OGRENMESINDE-TEORIDEN-ORNEK-MATLAB-UYGULAMALARINA-KADAR-DESTEK-VEKTOR-MAKINELERI.pdf


# In[ ]:





# In[ ]:





# In[ ]:





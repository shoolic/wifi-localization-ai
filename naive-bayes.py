from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np

data = np.loadtxt('wifi_localization.txt', delimiter='\t')
X = data[:, 0:-1]  # select columns 0 through end-1
y = data[:, -1]   # select last column, the room id

gnb = GaussianNB()
scores = []
cv_size = []

for i in range(10,500,50):
    cv_size.append(i)
    scores.append((cross_val_score(gnb, X, y, cv=i)).mean())

print(scores)
joblib.dump(gnb, 'wifi-localizaition-model.pkl', compress=9)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

cancer = load_breast_cancer()

a = np.random.randint(10)
b = np.random.randint(10)



df_features_all = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

df_features = df_features_all.iloc[:,[6,9]]
#df_features = df_features_all.iloc[:,[a,b]]


df_target = pd.DataFrame(cancer['target'], columns = ['Cancer'])

print(df_features.head(2))



X_train, X_test, y_train, y_test = train_test_split(df_features, np.ravel(df_target), test_size=0.30, random_state=101)



model = SVC()


model.fit(X_train, y_train)

predictions = model.predict(X_test)



print(classification_report(y_test, predictions))



param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}



grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)

grid.fit(X_train, y_train)

print('\n')
print('The best parameters are ', grid.best_params_)


grid_predictions = grid.predict(X_test)


print(classification_report(y_test, grid_predictions))





fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)


metrics.plot_roc_curve(model, X_test, y_test)


auc2 = metrics.auc(fpr, tpr)
auc = metrics.roc_auc_score(y_test, predictions)






X_test=np.array(X_test)
y_test=np.array(y_test)
plt.figure()
plot_decision_regions(X_test, y_test, clf=model, legend=2)
plt.xlim([min(X_train.iloc[:,0]), max(X_train.iloc[:,0])])
plt.ylim([min(X_train.iloc[:,1]), max(X_train.iloc[:,1])])
plt.show()
plt.close()







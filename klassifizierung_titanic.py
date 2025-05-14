from analyse_titanic import x_test, x_train, x_vali, y_test, y_train, y_vali
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

PARAMS_GRID = {"max_depth": [2, 3, 4, 5, 6],
               "min_samples_split": [3, 4, 5, 6, 7, 8],
                "min_samples_leaf": [3, 5, 7, 9]}

cls = DecisionTreeClassifier()
cv = RandomizedSearchCV(cls, param_distributions=PARAMS_GRID, cv=3, scoring="accuracy", verbose=2, n_jobs=-1, n_iter=100)
cv.fit(x_train, y_train)

cls = cv.best_estimator_

y_train_pred = cls.predict(x_train)

accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy auf Train: ", accuracy_train)

y_vali_pred = cls.predict(x_vali)
accuracy_vali = accuracy_score(y_vali, y_vali_pred)
print("Accuracy auf Vali: ", accuracy_vali)

print("Overfitting: ", accuracy_train - accuracy_vali)

df = pd.DataFrame({"feature": x_train.columns,
                   "score": cls.feature_importances_})

# print(df)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(cls,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=15)
# plt.show(block=True)
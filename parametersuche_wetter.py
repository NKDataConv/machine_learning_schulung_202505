from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali, x_test, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
from scipy.stats import randint as sp_randint

# Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100],
#     'max_depth': [3, 5, 7]
# }

param_dist = {"max_depth": sp_randint(3, 8),
              "n_estimators": sp_randint(80, 120)}

# Initialize the classifier
cls = GradientBoostingClassifier()

random_search = RandomizedSearchCV(estimator=cls,
                                   param_distributions=param_dist,
                                   n_iter=2,  # Number of parameter settings to sample
                                   scoring='accuracy',
                                   cv=2,
                                   random_state=42,
                                   verbose=2)
random_search.fit(x_train, y_train)
cls = random_search.best_estimator_

# Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=cls, param_grid=param_grid,
#                            scoring='accuracy', n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)
#
# cls = grid_search.best_estimator_

###### Prediction on Train Set #####
y_train_pred = cls.predict(x_train)

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

accuracy_train = accuracy_score(df["actual"], df["prediction"])
precision_train = precision_score(df["actual"], df["prediction"])
recall_train = recall_score(df["actual"], df["prediction"])
print("Accuracy auf Train: ", accuracy_train)
print("Precision auf Train: ", precision_train)
print("Recall auf Train: ", recall_train)


##### Prediction on Validation Set #####
y_vali_pred = cls.predict(x_vali)

df_vali = pd.DataFrame({'actual': y_vali,
                'prediction': y_vali_pred})

accuracy_vali = accuracy_score(df_vali["actual"], df_vali["prediction"])
precision_vali = precision_score(df_vali["actual"], df_vali["prediction"])
recall_vali = recall_score(df_vali["actual"], df_vali["prediction"])
print("Accuracy auf Vali: ", accuracy_vali)
print("Precision auf Vali: ", precision_vali)
print("Recall auf Vali: ", recall_vali)

overfitting = accuracy_train - accuracy_vali
overfitting_recall = recall_train - recall_vali
print("Overfitting Accuracy: ", overfitting)
print("Overfitting Recall: ", overfitting_recall)


##### Prediction on Test Set #####
y_test_pred = cls.predict(x_test)

df_test = pd.DataFrame({'actual': y_test,
                'prediction': y_test_pred})

accuracy_test = accuracy_score(df_test["actual"], df_test["prediction"])
precision_test = precision_score(df_test["actual"], df_test["prediction"])
recall_test = recall_score(df_test["actual"], df_test["prediction"])
print("Accuracy auf Test: ", accuracy_test)
print("Precision auf Test: ", precision_test)
print("Recall auf Test: ", recall_test)

print("=" * 50)
print("Information Leakage:", accuracy_vali - accuracy_test)

###### Feature Importances ######
# df_feature_importance = pd.DataFrame({"score": cls.feature_importances_,
#                                       "feature": x_train.columns})
#
# df_feature_importance.sort_values(by="score", ascending=False)
# print(df_feature_importance)

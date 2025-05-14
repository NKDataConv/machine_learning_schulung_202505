from datenaufbereitung_wetter import x_train, x_vali, y_risk_mm_train, y_risk_mm_vali
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {"max_depth": [3, 4, 5],
            "min_samples_split": [10, 20, 30, 40, 50, 60]}
reg = DecisionTreeRegressor()

gs = GridSearchCV(reg, param_grid=PARAM_GRID, cv=3, scoring="neg_mean_squared_error", verbose=2)
gs.fit(x_train, y_risk_mm_train)

reg = gs.best_estimator_
reg.fit(x_train, y_risk_mm_train)
y_risk_mm_train_pred = reg.predict(x_train)

mse = mean_squared_error(y_risk_mm_train_pred, y_risk_mm_train)
mae = mean_absolute_error(y_risk_mm_train_pred, y_risk_mm_train)
print("MSE auf Train: ", mse)
print("MAE auf Train: ", mae)

y_risk_mm_vali_pred = reg.predict(x_vali)

mse = mean_squared_error(y_risk_mm_vali_pred, y_risk_mm_vali)
mae = mean_absolute_error(y_risk_mm_vali_pred, y_risk_mm_vali)
print("MSE auf Vali: ", mse)
print("MAE auf Vali: ", mae)

import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_risk_mm_vali,
                         y=y_risk_mm_vali_pred,
                         mode="markers",
                         name="Prediction"))
fig.show()

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(reg,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=10)
# plt.show(block=True)

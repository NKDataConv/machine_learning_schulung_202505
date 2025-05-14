from datenaufbereitung_wetter import x_train, x_vali, y_risk_mm_train, y_risk_mm_vali
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

reg = DecisionTreeRegressor(max_depth=3)

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

print(y_risk_mm_vali_pred)
print(set(y_risk_mm_vali_pred))

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(reg,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=10)
# plt.show(block=True)

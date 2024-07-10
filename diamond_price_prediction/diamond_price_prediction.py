from tkinter import *
import tkinter as tk
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
import lightgbm as lgbm
import xgboost as xgbt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,train_test_split, KFold
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import math
import seaborn as sns
import joblib
import json
import time

# Đọc dữ liệu
data = pd.read_csv('D:/Graduate_Dessertation/diamonds1.csv')

numeric_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Tạo một bản sao của dữ liệu gốc để làm sạch
data_cleaned = data.copy()

#lựa chọn thuộc tính
#Xóa cột Unnamed(STT)
data_cleaned = data_cleaned.drop(["Unnamed: 0"], axis=1)
data_cleaned.shape

#kiểm tra dữ liệu thiếu và các biến phân loại
data_cleaned.info()

# Phát hiện và xử lý các giá trị ngoại lai cho mỗi cột số liệu
for column in numeric_columns:
    Q1 = data_cleaned[column].quantile(0.25)
    Q3 = data_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Loại bỏ các mẫu có giá trị ngoại lai trong cột hiện tại
    data_cleaned = data_cleaned[(data_cleaned[column] >= lower_bound) & (data_cleaned[column] <= upper_bound)]
 
# Hiển thị số lượng hàng bị loại bỏ
print(f"Số lượng mẫu còn lại sau khi loại bỏ các giá trị ngoại lai: {len(data_cleaned)}")
print(f"Số lượng mẫu ban đầu: {len(data)}")
print(f"Số lượng mẫu bị loại bỏ: {len(data) - len(data_cleaned)}")
print(data_cleaned.describe())

#các biểu đồ boxplot sau khi xử lý ngoại lai
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data_cleaned[column])
    plt.title('Boxplot of ' + str(column))
    plt.show()
 
#Mã hóa các cột phân loại
label_encoders = {}
for column in ['cut', 'color', 'clarity']:
    le = LabelEncoder()
    data_cleaned.loc[:, column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

#biến đổi logarit
data_cleaned['price'] = np.log(data_cleaned['price'] + 1)
print(data_cleaned.head())

#Chuẩn hóa tất cả các cột
columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
scaler = StandardScaler()
data_cleaned[columns] = scaler.fit_transform(data_cleaned[columns])
print(data_cleaned.head())

#Dữ liệu sạch
X_scaled = data_cleaned.drop(["price"],axis =1)
y = data_cleaned['price']

#============================================================================================================================
#Linear Regression
#Huấn luyện mô hình Linear Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lr = LinearRegression()

# # Tính thời gian bắt đầu
# start_time_lr = time.time()

# Áp dụng cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_score_lr = cross_val_score(lr, X_train_lr, y_train_lr, cv=kf, scoring='neg_mean_squared_error')

lr.fit(X_train_lr, y_train_lr)
y_pred_lr = lr.predict(X_test_lr)

# # Tính thời gian kết thúc
# end_time_lr = time.time()
# running_time_lr = end_time_lr - start_time_lr

#metrics
r2_score_lr = r2_score(y_test_lr, y_pred_lr)
mse_lr = -1*(cv_score_lr.mean())
mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

#==============================================================================================================
#Decision Tree
#Huấn luyện mô hình Decision Tree
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Áp dụng Bayesian Optimization tìm siêu tham số
# # Hàm đánh giá
# def dt_evaluate(splitter, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha):
#     params = {
#         'criterion': 'squared_error',
#         'splitter': 'best' if splitter < 0.5 else 'random',
#         'max_depth': int(max_depth),
#         'min_samples_split': int(min_samples_split),
#         'min_samples_leaf': int(min_samples_leaf),
#         'max_features': None if max_features > 0.99 else max_features,
#         'max_leaf_nodes': int(max_leaf_nodes),
#         'min_impurity_decrease': min_impurity_decrease,
#         'ccp_alpha': ccp_alpha
#     }
#     # Sử dụng cross-validation để đánh giá mô hình
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_val_score(DecisionTreeRegressor(**params), X_train_dt, y_train_dt, scoring='neg_mean_squared_error', cv=kf)
#     return cv_results.mean()
# # Đặt giới hạn cho các tham số
# param_bounds = {
#     'splitter': (0, 1),  # 0 for 'best', 1 for 'random'
#     'max_depth': (1, 20),
#     'min_samples_split': (2, 20),
#     'min_samples_leaf': (1, 20),
#     'max_features': (0.1, 1.0),
#     'max_leaf_nodes': (2, 100),  
#     'min_impurity_decrease': (0.0, 1.0),
#     'ccp_alpha': (0.0, 0.1)
# }
# # Khởi tạo Bayesian Optimization
# optimizer = BayesianOptimization(
#     f=dt_evaluate,
#     pbounds=param_bounds,
#     random_state=42,
#     verbose=2
# )

# # Tính thời gian bắt đầu
# start_time_dt = time.time()
# # Thực hiện tối ưu hóa
# optimizer.maximize(init_points=10, n_iter=50)

# # Lấy các tham số tốt nhất
# best_params_dt = optimizer.max['params']
# best_params_dt['splitter'] = 'best' if best_params_dt['splitter'] < 0.5 else 'random'
# best_params_dt['max_depth'] = int(best_params_dt['max_depth'])
# best_params_dt['min_samples_split'] = int(best_params_dt['min_samples_split'])
# best_params_dt['min_samples_leaf'] = int(best_params_dt['min_samples_leaf'])
# best_params_dt['max_leaf_nodes'] = int(best_params_dt['max_leaf_nodes'])
# best_params_dt['max_features'] = None if best_params_dt['max_features'] > 0.99 else best_params_dt['max_features']

# print("Các tham số tốt nhất tìm được cho Decision Tree: ", best_params_dt)
best_params_dt = {
        'criterion': 'squared_error',
        'splitter': 'best',
        'max_depth': 18,
        'min_samples_split': 4,
        'min_samples_leaf': 7,
        'max_features': None,
        'max_leaf_nodes': 32,
        'min_impurity_decrease': 0.0,
        'ccp_alpha': 0.0
    }
# Huấn luyện mô hình với các tham số tốt nhất
dt = DecisionTreeRegressor(**best_params_dt)
dt.fit(X_train_dt, y_train_dt)
y_pred_dt = dt.predict(X_test_dt)

# # Tính thời gian kết thúc
# end_time_dt = time.time()
# running_time_dt = end_time_dt - start_time_dt

#metrics
r2_score_dt = r2_score(y_test_dt, y_pred_dt)
mse_dt = mean_squared_error(y_test_dt, y_pred_dt)
mae_dt = mean_absolute_error(y_test_dt, y_pred_dt)
rmse_dt = root_mean_squared_error(y_test_dt, y_pred_dt)

#==================================================================================================================
#Random Forest
#Huấn luyện mô hình Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Áp dụng Bayesian Optimization tìm siêu tham số
# # Hàm đánh giá
# def rf_evaluate(max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap):
#     params = {
#         'criterion': 'squared_error',
#         #'n_estimators': int(n_estimators),
#         'n_estimators': 200,
#         'max_depth': int(max_depth),
#         'min_samples_split': int(min_samples_split),
#         'min_samples_leaf': int(min_samples_leaf),
#         'max_features': None if max_features > 0.99 else max_features,
#         'bootstrap': bool(int(bootstrap))
#     }
#     # Sử dụng cross-validation để đánh giá mô hình
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_val_score(RandomForestRegressor(**params), X_train_rf, y_train_rf, scoring='neg_mean_squared_error', cv=kf)
#     return cv_results.mean()
# # Đặt giới hạn cho các tham số
# param_bounds = {
#     #'n_estimators': (50, 200),
#     'max_depth': (1, 20),
#     'min_samples_split': (2, 10),
#     'min_samples_leaf': (1, 10),
#     'max_features': (0.1, 1.0),
#     'bootstrap': (0, 1)  # Sử dụng 0 cho False và 1 cho True
# }
# # Khởi tạo Bayesian Optimization
# optimizer = BayesianOptimization(
#     f=rf_evaluate,
#     pbounds=param_bounds,
#     random_state=42,
#     verbose=2
# )

# # Tính thời gian bắt đầu
# start_time_rf = time.time()

# # Thực hiện tối ưu hóa
# optimizer.maximize(init_points=10, n_iter=50)
# # Lấy các tham số tốt nhất
# best_params_rf = optimizer.max['params']
# #best_params_rf['n_estimators'] = int(best_params_rf['n_estimators'])
# best_params_rf['max_depth'] = int(best_params_rf['max_depth'])
# best_params_rf['min_samples_split'] = int(best_params_rf['min_samples_split'])
# best_params_rf['min_samples_leaf'] = int(best_params_rf['min_samples_leaf'])
# best_params_rf['max_features'] = None if best_params_rf['max_features'] > 0.99 else best_params_rf['max_features']
# best_params_rf['bootstrap'] = bool(int(best_params_rf['bootstrap']))

# print("Các tham số tốt nhất tìm được cho Random Forest: ", best_params_rf)
best_params_rf = {
        'criterion': 'squared_error',
        'n_estimators': 300,
        'max_depth': 18,
        'min_samples_split': 4,
        'min_samples_leaf': 1,
        'max_features': None,
        'bootstrap': True
    }
# Huấn luyện mô hình với các tham số tốt nhất
rf = RandomForestRegressor(**best_params_rf)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test_rf)

# # Tính thời gian kết thúc
# end_time_rf = time.time()
# running_time_rf = end_time_rf - start_time_rf

#metric
r2_score_rf = r2_score(y_test_rf, y_pred_rf)
mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
rmse_rf = root_mean_squared_error(y_test_rf, y_pred_rf)

#==============================================================================
#XGBoost
#Huấn luyện mô hình Random Forest
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# # Áp dụng bayesian Optimization tìm siêu tham số
# # Hàm đánh giá
# def xgb_evaluate(colsample_bylevel, colsample_bynode, colsample_bytree, gamma, learning_rate, max_depth, min_child_weight, reg_alpha, subsample):
#     params = {
#         'objective': 'reg:squarederror',
#         'colsample_bylevel': colsample_bylevel,
#         'colsample_bynode': colsample_bynode,
#         'colsample_bytree': colsample_bytree,
#         'gamma': gamma,
#         'learning_rate': learning_rate,
#         'max_depth': int(max_depth),
#         'min_child_weight': min_child_weight,
#         #'n_estimators': int(n_estimators),
#         'n_estimators': 200,
#         'reg_alpha': reg_alpha,
#         'subsample': subsample
#     }
#     # Sử dụng cross-validation để đánh giá mô hình
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_val_score(xgbt.XGBRegressor(**params), X_train_xgb, y_train_xgb, scoring='neg_mean_squared_error', cv=kf)
#     return cv_results.mean()
# # Đặt giới hạn cho các tham số
# param_bounds = {
#     'colsample_bylevel': (0.1, 1.0),
#     'colsample_bynode': (0.1, 1.0),
#     'colsample_bytree': (0.1, 1.0),
#     'gamma': (0, 5),
#     'learning_rate': (0.01, 0.3),
#     'max_depth': (3, 10),
#     'min_child_weight': (1, 10),
#     #'n_estimators': (50, 200),
#     'reg_alpha': (0, 1),
#     'subsample': (0.5, 1.0)
# }
# # Khởi tạo Bayesian Optimization
# optimizer = BayesianOptimization(
#     f=xgb_evaluate,
#     pbounds=param_bounds,
#     random_state=42,
#     verbose=2
# )

# # Tính thời gian bắt đầu
# start_time_xgb = time.time()

# # Thực hiện tối ưu hóa
# optimizer.maximize(init_points=10, n_iter=50)
# # Lấy các tham số tốt nhất
# best_params_xgb = optimizer.max['params']
# best_params_xgb['max_depth'] = int(best_params_xgb['max_depth'])
# #best_params_xgb['n_estimators'] = int(best_params_xgb['n_estimators'])

# print("Các tham số tốt nhất tìm được cho XGBoost: ", best_params_xgb)

best_params_xgb = {
        'objective': 'reg:squarederror',
        'colsample_bylevel': 0.6202021330485812,
        'colsample_bynode': 0.5464853306416603,
        'colsample_bytree': 0.7981893683173839,
        'gamma': 0.29457096262183624,
        'learning_rate': 0.09298914212723708,
        'max_depth': 6,
        'min_child_weight': 1.1881223044998184,
        'n_estimators': 300,
        'reg_alpha': 0.9900953483578221,
        'subsample': 0.9755613131973471     
    }
# Huấn luyện mô hình với các tham số tốt nhất
xgb = xgbt.XGBRegressor(**best_params_xgb)
xgb.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = xgb.predict(X_test_xgb)

# # Tính thời gian kết thúc
# end_time_xgb = time.time()
# running_time_xgb = end_time_xgb - start_time_xgb

#metrics
r2_score_xgb = r2_score(y_test_xgb, y_pred_xgb)
mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_xgb, y_pred_xgb)
rmse_xgb = root_mean_squared_error(y_test_xgb, y_pred_xgb)

#==============================================================================================================================
#LightGBM
#Huấn luyện mô hình LightGBM
X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# # Áp dụng Bayesian optimization tìm siêu tham số
# # Hàm đánh giá
# def gbm_evaluate(colsample_bytree, learning_rate, max_depth, min_child_samples, min_child_weight, min_split_gain, num_leaves, reg_alpha, reg_lambda, subsample):
#     params = {
#         'objective': 'regression',
#         'metric': 'mse',
#         'data_sample_strategy':'goss',
#         'boosting_type': 'gbdt',
#         'colsample_bytree': colsample_bytree,
#         'learning_rate': learning_rate,
#         'max_depth': int(max_depth),
#         'min_child_samples': int(min_child_samples),
#         'min_child_weight': min_child_weight,
#         'min_split_gain': min_split_gain,
#         #'n_estimators': int(n_estimators),
#         'n_estimators': 200,
#         'num_leaves': int(num_leaves),
#         'reg_alpha': reg_alpha,
#         'reg_lambda': reg_lambda,
#         'subsample': subsample,
#         'force_col_wise': 'true'
#     }
#     # Sử dụng cross-validation để đánh giá mô hình
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_val_score(lgbm.LGBMRegressor(**params), X_train_gbm, y_train_gbm, scoring='neg_mean_squared_error', cv=kf)
#     return cv_results.mean()

# # Đặt giới hạn cho các tham số
# param_bounds = {
#     'colsample_bytree': (0.1, 1.0),
#     'learning_rate': (0.01, 0.3),
#     'max_depth': (1, 10),
#     'min_child_samples': (5, 100),
#     'min_child_weight': (0.001, 10.0),
#     'min_split_gain': (0.0, 1.0),
#     #'n_estimators': (50, 200),
#     'num_leaves': (20, 200),
#     'reg_alpha': (0.0, 1.0),
#     'reg_lambda': (0.0, 1.0),
#     'subsample': (0.5, 1.0)
# }
# # Khởi tạo Bayesian Optimization
# optimizer = BayesianOptimization(
#     f=gbm_evaluate,
#     pbounds=param_bounds,
#     random_state=42,
#     verbose=2
# )
# # Tính thời gian bắt đầu
# start_time_gbm = time.time()

# # Thực hiện tối ưu hóa
# optimizer.maximize(init_points=10, n_iter=50)
# # Lấy các tham số tốt nhất
# best_params_gbm = optimizer.max['params']
# best_params_gbm['max_depth'] = int(best_params_gbm['max_depth'])
# best_params_gbm['min_child_samples'] = int(best_params_gbm['min_child_samples'])
# #best_params_gbm['n_estimators'] = int(best_params_gbm['n_estimators'])
# best_params_gbm['num_leaves'] = int(best_params_gbm['num_leaves'])

# print("Các tham số tốt nhất tìm được cho LightGBM: ", best_params_gbm)

best_params_gbm = {
        'objective': 'regression',
        'metric': 'mse',
        'data_sample_strategy':'goss',
        'boosting_type': 'gbdt',
        'colsample_bytree': 0.4008842459170925,
        'learning_rate': 0.17792424177857247,
        'max_depth': 9,
        'min_child_samples': 77,
        'min_child_weight': 9.624284137560164,
        'min_split_gain': 0.004755464327775183,
        'n_estimators': 300,
        'num_leaves': 51,
        'reg_alpha': 0.46627814981780924,
        'reg_lambda': 0.1937006096906072,
        'subsample': 0.9168732885184592
}
# Huấn luyện mô hình với các tham số tốt nhất
gbm = lgbm.LGBMRegressor(**best_params_gbm)
gbm.fit(X_train_gbm, y_train_gbm)
y_pred_gbm = gbm.predict(X_test_gbm)

# # Tính thời gian kết thúc
# end_time_gbm = time.time()
# running_time_gbm = end_time_gbm - start_time_gbm

#metrics
r2_score_gbm = r2_score(y_test_gbm, y_pred_gbm)
mse_gbm = mean_squared_error(y_test_gbm, y_pred_gbm)
mae_gbm = mean_absolute_error(y_test_gbm, y_pred_gbm)
rmse_gbm = root_mean_squared_error(y_test_gbm, y_pred_gbm)

#===================================================================================
# Lưu scaler, mô hình và các label encoders
joblib.dump(scaler, 'scaler.pkl')
for column, le in label_encoders.items():
    joblib.dump(le, f'le_{column}.pkl')
# Tải scaler, mô hình và các label encoders đã lưu
scaler = joblib.load('scaler.pkl')
le_cut = joblib.load('le_cut.pkl')
le_color = joblib.load('le_color.pkl')
le_clarity = joblib.load('le_clarity.pkl')
#=========================================================================================
# Tạo ứng dụng tkinter
form = Tk()
form.configure(bg='#fff')
form.title("Dự đoán giá trị kim cương")
form.iconbitmap('D:/Graduate_Dessertation/icon.ico')
form.state('zoomed')
#form.attributes('-fullscreen', True)

# Tạo Canvas để chứa Frame cuộn
canvas = tk.Canvas(form)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.config(yscrollcommand=scrollbar.set)

scrollable_frame = tk.Frame(canvas)
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
scrollable_frame.bind("<Configure>", on_frame_configure)

# Thêm frame vào trong canvas bằng window_create
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

frame00 = Frame(scrollable_frame)
frame00.pack(side=TOP,fill="x")
label001 = Label(frame00, text="Các siêu tham số tốt nhất tìm được",font=("Arial Bold", 14), fg="red")
label001.pack(side=LEFT, padx=10, pady=10)

# label06 = Label(frame01,font=("Arial Bold", 13),justify="left")
# label06.pack(side=LEFT, padx=10, pady=10)
# label06.configure(text="Thời gian chạy của mô hình Linear Regression: %.2f" % running_time_lr + "giây" + '\n'
#                        +"Thời gian chạy của mô hình Decision Tree: %.2f" % running_time_dt + "giây" + '\n'
#                        +"Thời gian chạy của mô hình Random Forest: %.2f" % running_time_rf + "giây" + '\n'
#                        +"Thời gian chạy của mô hình XGBoost: %.2f" % running_time_xgb + "giây" + '\n'
#                        +"Thời gian chạy của mô hình LightGBM: %.2f" % running_time_gbm + "giây")

#chuyển các tham số tìm được từ dạng dict sang dạng string
str_best_params_dt= json.dumps(best_params_dt, indent=4)
str_best_params_rf= json.dumps(best_params_rf, indent=4)
str_best_params_xgb= json.dumps(best_params_xgb, indent=4)
str_best_params_gbm= json.dumps(best_params_gbm, indent=4)

frame03 = Frame(scrollable_frame)
frame03.pack(side=TOP,fill="x")
label02 = Label(frame03,font=("Arial Bold", 13),justify="left")
label03 = Label(frame03,font=("Arial Bold", 13),justify="left")
label02.pack(fill="x", side=LEFT, padx=10, pady=10)
label03.pack(fill="x", side=LEFT, padx=30, pady=20)
label02.configure(text="Các tham số tốt nhất tìm được cho mô hình Decision Tree: " + str(str_best_params_dt))
label03.configure(text="Các tham số tốt nhất tìm được cho mô hình Random Forest: " + str(str_best_params_rf))

frame04 = Frame(scrollable_frame)
frame04.pack(side=TOP,fill="x")
label04 = Label(frame04,font=("Arial Bold", 13),justify="left")
label05 = Label(frame04,font=("Arial Bold", 13),justify="left")
label04.pack(fill="x", side=LEFT, padx=10, pady=10)
label05.pack(fill="x", side=LEFT, padx=35, pady=20)
label04.configure(text="Các tham số tốt nhất tìm được cho mô hình XGBoost: " + str(str_best_params_xgb))
label05.configure(text="Các tham số tốt nhất tìm được cho mô hình LightGBM: " + str(str_best_params_gbm))

frame02 = Frame(scrollable_frame)
frame02.pack(side=TOP,fill="x")
label021 = Label(frame02, text="Nhập các thông tin kim cương",font=("Arial Bold", 14), fg="red")
label021.pack(side=LEFT, padx=10, pady=10)

# Tạo các nhãn và text box cho các đặc trưng
labels = ['Carat', 'Depth', 'Table', 'X', 'Y', 'Z']
entries = {}
for label in labels:
        frame = Frame(scrollable_frame)
        frame.pack(side=TOP,fill="x")

        lbl = Label(frame, text=label, width=10)
        lbl.pack(side=LEFT, padx = 10, pady = 10)
        
        entry = Entry(frame, width=23)
        entry.pack(side=LEFT,fill="x", padx = 10, pady = 10)
        entries[label.lower()] = entry

# Tạo các combobox cho các cột phân loại
categorical_labels = {
    'cut': list(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']),
    'color': list(['J', 'I', 'H', 'G', 'F', 'E', 'D']),
    'clarity': list(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
}
for label, values in categorical_labels.items():
        frame = Frame(scrollable_frame)
        frame.pack(side=TOP,fill="x")

        lbl = Label(frame, text=label.capitalize(), width=10)
        lbl.pack(side=LEFT, padx = 10, pady = 10)

        combobox = ttk.Combobox(frame, values=values)
        combobox.pack(side=LEFT,fill="x", padx = 10, pady = 10)
        entries[label.lower()] = combobox

# Hàm lấy giá trị 
def validation():
        # Lấy dữ liệu từ các text box và combobox
        input_data = {
            'carat': entries['carat'].get(),
            'cut': entries['cut'].get(),
            'color': entries['color'].get(),
            'clarity': entries['clarity'].get(),
            'depth': entries['depth'].get(),
            'table': entries['table'].get(),
            'x': entries['x'].get(),
            'y': entries['y'].get(),
            'z': entries['z'].get()
        }
        # Chuyển đổi giá trị đầu vào thành kiểu float
        input_data = {k: float(v) if k in ['carat', 'depth', 'table', 'x', 'y', 'z'] else v for k, v in input_data.items()}
        
        input_data['cut'] = le_cut.transform([input_data['cut']])[0]
        input_data['color'] = le_color.transform([input_data['color']])[0]
        input_data['clarity'] = le_clarity.transform([input_data['clarity']])[0]
        
        input_data_df = pd.DataFrame([input_data])
    
        # Chuẩn hóa các cột 
        input_data_scaled = scaler.transform(input_data_df)
        input_data_scaled = input_data_scaled.reshape(1,-1)
        if((input_data['carat'] == '') or (input_data['cut'] == '') or (input_data['color'] == '') or (input_data['clarity'] == '') or (input_data['depth'] == '') or (input_data['table'] == '') or (input_data['x'] == '') or (input_data['y'] == '') or (input_data['z'] == '')):
            return None
        else:
            return input_data_scaled, input_data_df.columns
#===========================================================================================================
#Các hàm thực thi thuật toán
#Dự đoán LinearRegression
def predict_linear_regression():
    try:
        input_data_scaled, feature_names = validation()
        if input_data_scaled is not None and feature_names is not None:
            try:
                input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_names)
                print(input_data_scaled_df)
                pred_lr = lr.predict(input_data_scaled_df)
                pred_lr = np.exp(pred_lr) - 1
                result_lr = np.round(pred_lr)
                label_lr.configure(text = str(result_lr))        
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
    except:
        messagebox.showinfo("Lỗi!", "Bạn cần nhập đầy đủ thông tin và đúng định dạng!")
#Dự đoán decision tree
def predict_decision_tree():
    try:
        input_data_scaled, feature_names = validation()
        if input_data_scaled is not None and feature_names is not None:
            try:
                input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_names)
                pred_dt = dt.predict(input_data_scaled_df)
                pred_dt = np.exp(pred_dt) - 1
                result_dt = np.round(pred_dt)
                label_dt.configure(text = str(result_dt))        
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
    except:
        messagebox.showinfo("Lỗi!", "Bạn cần nhập đầy đủ thông tin và đúng định dạng!")
#Dự đoán random forest
def predict_random_forest():
    try:
        input_data_scaled, feature_names = validation()
        if input_data_scaled is not None and feature_names is not None:
            try:
                input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_names)
                pred_rf = rf.predict(input_data_scaled_df)
                pred_rf = np.exp(pred_rf) - 1
                result_rf = np.round(pred_rf)
                label_rf.configure(text = str(result_rf))        
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
    except:
        messagebox.showinfo("Lỗi!", "Bạn cần nhập đầy đủ thông tin và đúng định dạng!")
#Dự đoán XGBoost
def predict_xgboost():
    try:
        input_data_scaled, feature_names = validation()
        if input_data_scaled is not None and feature_names is not None:
            try:
                input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_names)
                pred_xgb = xgb.predict(input_data_scaled_df)
                pred_xgb = np.exp(pred_xgb) - 1
                result_xgb = np.round(pred_xgb)
                label_xgb.configure(text = str(result_xgb))        
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
    except:
        messagebox.showinfo("Lỗi!", "Bạn cần nhập đầy đủ thông tin và đúng định dạng!")
#Dự đoán LightGBM
def predict_light_gbm():
    try:
        input_data_scaled, feature_names = validation()
        if input_data_scaled is not None and feature_names is not None:
            try:
                input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_names)
                pred_gbm = gbm.predict(input_data_scaled_df)
                pred_gbm = np.exp(pred_gbm) - 1
                result_gbm = np.round(pred_gbm)
                label_gbm.configure(text = str(result_gbm))        
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
    except:
        messagebox.showinfo("Lỗi!", "Bạn cần nhập đầy đủ thông tin và đúng định dạng!")
#=========================================================================================================================
#frame tiêu đề Các tiêu chí đánh giá
frame = Frame(scrollable_frame)
frame.pack(side=TOP,fill="x")
label1 = Label(frame, text="Các tiêu chí đánh giá của các thuật toán",font=("Arial Bold", 14), fg="red")
label1.pack(side=LEFT, padx=10, pady=10)

#Linear Regression
frame0 = Frame(scrollable_frame, width=100)
frame0.pack(side=TOP,fill="x")
label_metrics_lr = Label(frame0,font=("Arial Bold", 11),justify="left")
label_metrics_lr.pack(side=LEFT, padx = 10, pady = 10)
label_metrics_lr.configure(text="Mô hình Linear Regression: "+'\n'
                       +"R2 score: " + str(np.round(r2_score_lr,3)) +'\n'
                       +"Mean squared error: " + str(np.round(mse_lr, 3))+'\n'
                       +"Mean absolute error: " + str(np.round(mae_lr, 3))+'\n'    
                       +"Root Mean Squared Error: " + str(np.round(rmse_lr,3)))
#Decision Tree
label_metrics_dt = Label(frame0,font=("Arial Bold", 11),justify="left")
label_metrics_dt.pack(side=LEFT, padx = 10, pady = 10)
label_metrics_dt.configure(text="Mô hình Decision Tree: "+'\n'
                       +"R2 score: " + str(np.round(r2_score_dt,3)) +'\n'
                       +"Mean squared error: " + str(np.round(mse_dt, 3))+'\n'
                       +"Mean absolute error: " + str(np.round(mae_dt, 3))+'\n' 
                       +"Root Mean Squared Error: " + str(np.round(rmse_dt,3)))
#Random Forest
label_metrics_rf = Label(frame0,font=("Arial Bold", 11),justify="left")
label_metrics_rf.pack(side=LEFT, padx = 10, pady = 10)
label_metrics_rf.configure(text="Mô hình Random Forest: "+'\n'
                       +"R2 score: " + str(np.round(r2_score_rf,3)) +'\n'
                       +"Mean squared error: " + str(np.round(mse_rf, 3))+'\n'
                       +"Mean absolute error: " + str(np.round(mae_rf, 3))+'\n'    
                       +"Root Mean Squared Error: " + str(np.round(rmse_rf,3)))
#XGBoost
label_metrics_xgb = Label(frame0,font=("Arial Bold", 11),justify="left")
label_metrics_xgb.pack(side=LEFT, padx = 10, pady = 10)
label_metrics_xgb.configure(text="Mô hình XGBoost: "+'\n'
                       +"R2 score: " + str(np.round(r2_score_xgb,3))+'\n'
                       +"Mean squared error: " + str(np.round(mse_xgb,3))+'\n'
                       +"Mean absolute error: " + str(np.round(mae_xgb, 3))+'\n'     
                       +"Root Mean Squared Error: "+ str(np.round(rmse_xgb,3)))
#LightGBM
label_metrics_ = Label(frame0,font=("Arial Bold", 11),justify="left")
label_metrics_.pack(side=LEFT, padx = 10, pady = 10)
label_metrics_.configure(text="Mô hình LightGBM: "+'\n'
                       +"R2 score: " + str(np.round(r2_score_gbm,3))+'\n'
                       +"Mean squared error: " + str(np.round(mse_gbm,3))+'\n'
                       +"Mean absolute error: " + str(np.round(mae_gbm, 3))+'\n'  
                       +"Root Mean Squared Error: "+ str(np.round(rmse_gbm,3)))
#=====================================================================================================
#frame tiêu đề buttons
frame2 = Frame(scrollable_frame)
frame2.pack(side=TOP,fill="x")
label1 = Label(frame2, text="Các nút thực thi các thuật toán hồi quy",font=("Arial Bold", 13), fg="red")
label1.pack(side=LEFT, padx=10, pady=10)
#Buttons      
frame1 = Frame(scrollable_frame)
frame1.pack(side=TOP,fill="x")
#Button cho Linear Regression
btn_lr = Button(frame1, text = 'Linear Regression:',bg="lightblue", bd=4, command = predict_linear_regression,width = 20)
btn_lr.pack(side=LEFT, padx = 20, pady = 20)
label_lr= Label(frame1, text='...')
label_lr.pack(side=LEFT,fill="x", padx = 10, pady = 10)
#Button cho Decision Tree
btn_dt = Button(frame1, text = 'Decision Tree:',bg="lightblue", bd=4, command = predict_decision_tree,width = 20)
btn_dt.pack(side=LEFT, padx = 20, pady = 20)
label_dt= Label(frame1, text='...')
label_dt.pack(side=LEFT,fill="x", padx = 10, pady = 10)
#Button cho Random Forest
btn_rf = Button(frame1, text = 'Random Forest',bg="lightblue", bd=4, command = predict_random_forest,width = 20)
btn_rf.pack(side=LEFT, padx = 20, pady = 20)
label_rf= Label(frame1, text='...')
label_rf.pack(side=LEFT,fill="x", padx = 10, pady = 10)
#Button cho XGBoost
btn_xgb = Button(frame1, text = 'XGBoost:',bg="lightblue", bd=4, command = predict_xgboost,width = 20)
btn_xgb.pack(side=LEFT, padx = 20, pady = 20)
label_xgb= Label(frame1, text='...')
label_xgb.pack(side=LEFT,fill="x", padx = 10, pady = 10)
#Button cho LightGBM
btn_gbm = Button(frame1, text = 'LightGBM:',bg="lightblue", bd=4, command = predict_light_gbm,width = 20)
btn_gbm.pack(side=LEFT, padx = 20, pady = 20)
label_gbm= Label(frame1, text='...')
label_gbm.pack(side=LEFT,fill="x", padx = 10, pady = 10)

form.mainloop()


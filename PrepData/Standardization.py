import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


train = pd.read_csv('D:\\MachineLearning\\BTL\\data\\train.csv')
val   = pd.read_csv('D:\\MachineLearning\\BTL\\data\\val.csv')
test  = pd.read_csv('D:\\MachineLearning\\BTL\\data\\test.csv')

########################################## TÁCH X - y ###########################################

X_train = train.drop('NObeyesdad', axis=1)
y_train = train['NObeyesdad']

X_val = val.drop('NObeyesdad', axis=1)
y_val = val['NObeyesdad']

X_test = test.drop('NObeyesdad', axis=1)
y_test = test['NObeyesdad']

###################################### CHUẨN HÓA DỮ LIỆU ######################################

robust_cols   = ['Age', 'NCP']
standard_cols = ['Height', 'Weight', 'CH2O', 'FAF']

# Copy để tránh warning
X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

# ===================== SCALER =====================
robust_scaler   = RobustScaler()
standard_scaler = StandardScaler()

# ===================== FIT trên TRAIN =====================
X_train_scaled[robust_cols]   = robust_scaler.fit_transform(X_train[robust_cols])
X_train_scaled[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])

# ===================== TRANSFORM VAL =====================
X_val_scaled[robust_cols]   = robust_scaler.transform(X_val[robust_cols])
X_val_scaled[standard_cols] = standard_scaler.transform(X_val[standard_cols])

# ===================== TRANSFORM TEST =====================
X_test_scaled[robust_cols]   = robust_scaler.transform(X_test[robust_cols])
X_test_scaled[standard_cols] = standard_scaler.transform(X_test[standard_cols])

# ===================== KIỂM TRA =====================
print("=" * 55)
print("TRAIN SAU KHI CHUẨN HÓA:")
print("=" * 55)
print(X_train_scaled[robust_cols + standard_cols].describe().round(3))

# ===================== SAVE =====================
train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
val_df   = pd.concat([X_val_scaled,   y_val.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test_scaled,  y_test.reset_index(drop=True)], axis=1)

train_df.to_csv('D:\\MachineLearning\\BTL\\train_scaled.csv', index=False)
val_df.to_csv('D:\\MachineLearning\\BTL\\val_scaled.csv', index=False)
test_df.to_csv('D:\\MachineLearning\\BTL\\test_scaled.csv', index=False)

print("\n Đã lưu train/val/test scaled")

############################################################# kiểm tra độ tốt của dữ liệu đã chuẩn hóa #############################################################




# Gender: Feature, Categorical, "Gender"
# Age : Feature, Continuous, "Age"
# Height: Feature, Continuous
# Weight: Feature Continuous
# family_history_with_overweight: Feature, Binary, " Has a family member suffered or suffers from overweight? "

# FAVC : Feature, Binary, " Do you eat high caloric food frequently? "
# FCVC : Feature, Integer, " Do you usually eat vegetables in your meals? "
# NCP : Feature, Continuous, " How many main meals do you have daily? "
# CAEC : Feature, Categorical, " Do you eat any food between meals? "
# SMOKE : Feature, Binary, " Do you smoke? "
# CH2O: Feature, Continuous, " How much water do you drink daily? "
# SCC: Feature, Binary, " Do you monitor the calories you eat daily? "
# FAF: Feature, Continuous, " How often do you have physical activity? "
# TUE : Feature, Integer, " How much time do you use technological devices such as cell phone, videogames, television, computer and others? "


# Age và NCP có phân phối lệch (skewed) nên dùng RobustScaler để giảm ảnh hưởng của outliers
# Height, Weight, CH2O, FAF có phân phối gần chuẩn (normal) nên dùng StandardScaler để chuẩn hóa về mean=0 và std=1, giúp PCA/LDA hoạt động tốt hơn.





#PCA thì sau scale cần phải kiểm tra 
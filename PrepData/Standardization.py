import pandas as pd
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
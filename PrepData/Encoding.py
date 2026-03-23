import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('D:\\MachineLearning\\BTL\\ObesityDataSet_raw_and_data_sinthetic.csv')

############################################## SPLIT ##########################################
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

############################################## ENCODING ##########################################

# 1. Binary Encoding (dùng mapping thay LabelEncoder)
binary_cols = ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'Gender']

binary_mapping = {
    'yes': 1, 'no': 0,
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0
}

for col in binary_cols:
    X_train[col] = X_train[col].map(binary_mapping)
    X_val[col]   = X_val[col].map(binary_mapping)
    X_test[col]  = X_test[col].map(binary_mapping)

# 2. MTRANS (Ordinal)
mtrans_order = ['Automobile', 'Motorbike', 'Public_Transportation', 'Bike', 'Walking']

for data in [X_train, X_val, X_test]:
    data['MTRANS'] = pd.Categorical(
        data['MTRANS'], categories=mtrans_order, ordered=True
    ).codes

    # xử lý nếu có giá trị lạ
    data['MTRANS'].replace(-1, None, inplace=True)

# 3. CALC, CAEC (Ordinal)
ordinal_order = ['no', 'Sometimes', 'Frequently', 'Always']
oe = OrdinalEncoder(categories=[ordinal_order, ordinal_order])

X_train[['CALC', 'CAEC']] = oe.fit_transform(X_train[['CALC', 'CAEC']])
X_val[['CALC', 'CAEC']]   = oe.transform(X_val[['CALC', 'CAEC']])
X_test[['CALC', 'CAEC']]  = oe.transform(X_test[['CALC', 'CAEC']])

# 4. TARGET (Ordinal)
target_order = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

oe_target = OrdinalEncoder(categories=[target_order])

y_train = oe_target.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_val   = oe_target.transform(y_val.values.reshape(-1,1)).ravel()
y_test  = oe_target.transform(y_test.values.reshape(-1,1)).ravel()

############################################## SAVE ##########################################

train_df = pd.concat([X_train, pd.Series(y_train, name='NObeyesdad')], axis=1)
val_df   = pd.concat([X_val, pd.Series(y_val, name='NObeyesdad')], axis=1)
test_df  = pd.concat([X_test, pd.Series(y_test, name='NObeyesdad')], axis=1)

train_df.to_csv('D:\\MachineLearning\\BTL\\train.csv', index=False)
val_df.to_csv('D:\\MachineLearning\\BTL\\val.csv', index=False)
test_df.to_csv('D:\\MachineLearning\\BTL\\test.csv', index=False)
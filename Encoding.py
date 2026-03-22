import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.read_csv('D:\\MachineLearning\\BTL\\ObesityDataSet_raw_and_data_sinthetic.csv')
# 2. Binary Encoding
binary_cols = ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'Gender']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# 3. MTRANS → Ordinal theo mức độ vận động (thấp → cao)
mtrans_order = ['Automobile', 'Motorbike', 'Public_Transportation', 'Bike', 'Walking']
df['MTRANS'] = pd.Categorical(df['MTRANS'], categories=mtrans_order, ordered=True).codes

# 4. Ordinal Encoding: CALC, CAEC
ordinal_order = ['no', 'Sometimes', 'Frequently', 'Always']
oe = OrdinalEncoder(categories=[ordinal_order, ordinal_order])
df[['CALC', 'CAEC']] = oe.fit_transform(df[['CALC', 'CAEC']])

# 5. Target Encoding: NObeyesdad
target_order = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
oe_target = OrdinalEncoder(categories=[target_order])
df[['NObeyesdad']] = oe_target.fit_transform(df[['NObeyesdad']])

df.to_csv('D:\\MachineLearning\\BTL\\ObesityDataSet_encoded_v4.csv', index=False)
print("Đã lưu file: ObesityDataSet_encoded_v4.csv")
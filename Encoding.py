import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('D:\\MachineLearning\\BTL\\ObesityDataSet_raw_and_data_sinthetic.csv')

# chiến thuật xử lý nonnumeric data:

#Binary Encoding: Gender, family_history_with_overweight, FAVC, SMOKE, SCC    

binary_cols = ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'Gender']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

#One-Hot Encoding: MTRANS

df = pd.get_dummies(df, columns=['MTRANS'], prefix='MTRANS')

mtrans_cols = [col for col in df.columns if col.startswith('MTRANS_')]
df[mtrans_cols] = df[mtrans_cols].astype(int)

#Ordinal Encoding: CAEC, CALC, NObeyesdad

ordinal_order = ['no', 'Sometimes', 'Frequently', 'Always']
oe = OrdinalEncoder(categories=[ordinal_order, ordinal_order])
df[['CALC', 'CAEC']] = oe.fit_transform(df[['CALC', 'CAEC']])

Target_ordinal_order = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
oe = OrdinalEncoder(categories=[Target_ordinal_order])
df[['NObeyesdad']] = oe.fit_transform(df[['NObeyesdad']])



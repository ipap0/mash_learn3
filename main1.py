import sklearn.datasets
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ar2d_1 = [['aaa'], ['bbb'], ['bbb'], ['ddd'], ['aaa'], ['ccc'], ['bbb']]
ar2d_2 = [['aaa', 'MSK'], ['bbb', 'SPB'], ['bbb', 'VLD'], ['ddd', 'SPB'], ['aaa', 'SPB'], ['ccc', 'MSK'], ['bbb', 'VLD']]
print('Ordinal')
encoder1 = OrdinalEncoder()
encoder1.fit(ar2d_1)
result1 = encoder1.transform(ar2d_1)
print(result1)
# result2 = encoder1.transform(ar2d_2)    #Нельзя! уже запомнили, что в исходном массиве 1 столбец
# print(result2)
result2 = encoder1.fit_transform(ar2d_2)
print(result2)

print('OneHot')
encoder2 = OneHotEncoder()
result3 = encoder2.fit_transform(ar2d_1)
print(result3)
print(result3.toarray())

result4 = encoder2.fit_transform(ar2d_2)
print(result4)
print(result4.toarray())

print('work with DataFrames')

df1 = pd.DataFrame({'name':['aaa', 'ccc', 'ddd', 'bbb', 'aaa', 'bbb', 'aaa']})
print('Ordinal')
ord1 = encoder1.fit_transform(df1)
print(ord1)
encoded_df1 = pd.DataFrame(ord1, columns=encoder1.get_feature_names_out())
print(encoded_df1)
check_df1 = pd.concat([df1, encoded_df1], axis=1)
print(check_df1)

print('OneHot')
encoded_columns = encoder2.fit_transform(df1)
print(encoded_columns)
encoded_df1 = pd.DataFrame(encoded_columns.toarray(), columns=encoder2.get_feature_names_out())
print(encoded_df1)
print( pd.concat([df1, encoded_df1], axis=1))

print('transform work dataset')
encoder3 = OneHotEncoder(handle_unknown='ignore')
encoder3.fit(df1)
df_work = pd.DataFrame({'name':['ccc', 'ccc', 'bbb', 'xxx',  'ddd', 'yyy']})
encoded_columns_work = encoder3.transform(df_work)
encoded_work = pd.DataFrame(encoded_columns_work.toarray(), columns=encoder3.get_feature_names_out())
print(encoded_work)
print(pd.concat([df_work, encoded_work], axis=1))
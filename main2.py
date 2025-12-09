import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Animals.csv', delimiter=';', quotechar="'")
print('Исходник')
print(df)

X = df[['Animal', 'Color']]
y = df['Target_Type']
#кодируем независимые переменные
encoder_X = OneHotEncoder(handle_unknown='ignore')
encoded_X_columns = encoder_X.fit_transform(X, y)
print(encoded_X_columns)
X_enc = pd.DataFrame(encoded_X_columns.toarray(),columns= encoder_X.get_feature_names_out())
print(X_enc)
#кодируем целевую переменную
encoder_y = LabelEncoder()
encoded_y_column = encoder_y.fit_transform(y)
print(encoded_y_column)
y_enc = pd.Series(encoded_y_column, name=y.name)
print(y_enc)
print(pd.DataFrame(y_enc))

#попробуем обучить классификатор на исходных данных целиком
model_tree = DecisionTreeClassifier(max_depth=10, random_state=111)
model_tree.fit(X_enc, y_enc)

# допустим, на входе зверек
creature = pd.DataFrame({'Animal':['cat', 'cat'], 'Color':['black', 'white']})

creature_encoded_columns = encoder_X.transform(creature)
creature_encoded = pd.DataFrame(creature_encoded_columns.toarray(), columns=encoder_X.get_feature_names_out())
prediction1 = model_tree.predict(creature_encoded_columns)
print(f'prediction1 is {prediction1}')

prediction2 = model_tree.predict(creature_encoded)
print(f'prediction2 is {prediction2}')


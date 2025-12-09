import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import  Pipeline

#построить Конвейер обработки животных на основе Animals2.csv

df = pd.read_csv('Animals2.csv', delimiter=';', quotechar="'")
# ? Разделить исходные данные на learn и test

X_learn = df[['Weight', 'Animal', 'Color']]
y_learn = df['Target_Type']

# process = ColumnTransformer(
#     transformers=[
#         ('Weight', StandardScaler()),
#         ('Animal', OneHotEncoder(handle_unknown='ignore')),
#         ('Color',  OneHotEncoder(handle_unknown='ignore'))
#     ]
# )

process = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Weight']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Animal', 'Color'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', process),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

encoder_y = LabelEncoder()
encoded_y_column = encoder_y.fit_transform(y_learn)
encoded_y_learn = pd.Series(encoded_y_column, name=y_learn.name)

#запуск ОБУчения
pipeline.fit(X_learn, encoded_y_learn)


# выполнить предсказания для каких-то других данных
X_work = pd.DataFrame({'Weight': [1000], 'Animal': ['rat'], 'Color': ['black']})

predict_work = pipeline.predict(X_work)
print(predict_work)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')

print(df.head())

X = df[['glucose', 'bloodpressure']]

y = df['diabetes']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()

X_train_1 = sc.fit_transform(X_train_1)

X_test_1 = sc.fit_transform(X_test_1)

model_1 = GaussianNB()

model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test_1)

accuracy = accuracy_score(y_test_1, y_pred_1)

print(accuracy)

X = df[['glucose', 'bloodpressure']]

y = df['diabetes']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()

X_train_2 = sc.fit_transform(X_train_2)

X_test_2 = sc.fit_transform(X_test_2)

model_2 = LogisticRegression(random_state = 0)

model_2.fit(X_train_2, y_train_2)

y_pred_2 = model_2.predict(X_test_2)

accuracy = accuracy_score(y_test_2, y_pred_2)

print(accuracy)

df = pd.read_csv('income.csv')

print(df.head())

print(df.describe())

X = df[['age', 'hours-per-week', 'education-num', 'capital-gain', 'capital-loss']]

y = df['income']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()

X_train_1 = sc.fit_transform(X_train_1)

X_test_1 = sc.fit_transform(X_test_1)

model_1 = GaussianNB()

model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test_1)

accuracy = accuracy_score(y_test_1, y_pred_1)

print(accuracy)

X = df[['age', 'hours-per-week', 'education-num', 'capital-gain', 'capital-loss']]

y = df['income']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()

X_train_2 = sc.fit_transform(X_train_2)

X_test_2 = sc.fit_transform(X_test_2)

model_2 = LogisticRegression(random_state = 0)

model_2.fit(X_train_2, y_train_2)

y_pred_2 = model_2.predict(X_test_2)

accuracy = accuracy_score(y_test_2, y_pred_2)

print(accuracy)
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 

data = pd.read_csv('data/train_and_test2.csv')

data = data[['Pclass', 'Sex', 'Age', '2urvived']].dropna()

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

X = data[['Pclass', 'Sex', 'Age']] # features
y = data['2urvived'] # target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# print(model.score(x_test, y_test))
cl = input('Enter the class: ')
se = input('Enter the sex: ')
ag = input('Enter the age: ')

new_passenger = pd.DataFrame([[cl, se , ag]], columns=['Pclass', 'Sex', 'Age'])
predi = model.predict(new_passenger)

print('Did survive: ', 'Yes' if predi[0] == 1 else 'No')
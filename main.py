# https://www.codeit.kr/learn/courses/machine-learning/3132
# 07. 랜덤 포레스트로 악성/양성 유방암 분류하기

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)

# 출력 코드
print(predictions)
print(score)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("/content/Cancer_Data.csv")


X = data[["radius_mean", "texture_mean", "perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean"]]

y = data["diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)  


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = [[17, 10, 122, 1001, 0.1184, 0.2777, 0.3001]]


prediction = model.predict(new_data)
print("Predicted:", prediction[0])


if prediction[0] == 0:
    print("Benign cancer")
else:
    print("Malignant cancer")

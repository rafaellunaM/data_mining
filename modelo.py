import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carregar o dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Salvar o modelo treinado em um arquivo .pkl
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo treinado e salvo com sucesso!")

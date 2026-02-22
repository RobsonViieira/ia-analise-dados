import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ===============================
# 1. Carregar os dados
# ===============================

# Lê o arquivo CSV
dados = pd.read_csv("data/dados.csv")

print("Dados carregados:")
print(dados)


# ===============================
# 2. Separar variáveis
# ===============================

# X = entrada (mes)
X = dados[["mes"]]

# y = saída (valor)
y = dados["valor"]


# ===============================
# 3. Dividir treino e teste
# ===============================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 4. Criar e treinar modelo
# ===============================

modelo = LinearRegression()

modelo.fit(X_treino, y_treino)


# ===============================
# 5. Fazer previsões
# ===============================

previsoes = modelo.predict(X_teste)


# ===============================
# 6. Mostrar resultados
# ===============================

print("\nPrevisões:")
for real, previsto in zip(y_teste, previsoes):
    print(f"Real: {real:.2f} | Previsto: {previsto:.2f}")


# ===============================
# 7. Visualização
# ===============================

plt.scatter(X, y)
plt.plot(X, modelo.predict(X), linestyle='--')
plt.xlabel("Mês")
plt.ylabel("Valor")
plt.title("Análise e Previsão de Dados")

plt.show()
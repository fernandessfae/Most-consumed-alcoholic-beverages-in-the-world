import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

'''Criação de correlação entre o consumo total de cada bebida alcoólica consumida VS
   total de álcool ingerido em todo o mundo, com exceções, per capita'''

#Carregamento da base de dados de todos os países, e remoção dos países não alcoólicos
bebida_mundo = pd.read_csv('Consumo de bebidas alcoólicas ao redor do mundo.csv')
for index, row in bebida_mundo.iterrows():
    if row['total_litres_of_pure_alcohol'] == 0:
        bebida_mundo = bebida_mundo.drop(index)

#Seleciona as colunas para serem as variáveis explanatória e resposta, respectivamente
X = bebida_mundo.iloc[:, 1:4]
y = bebida_mundo.iloc[:, 4]

#Criação do modelo de regressão linear múltipla
modelo = LinearRegression()
modelo.fit(X, y)

# A função score serve para mostrar o grau de acurácia do modelo criado.
modelo.score(X, y)

# Função de mostrar todos os valores detalhados, parecido no R
modelo_ajustado = sm.ols(formula = 'total_litres_of_pure_alcohol ~ beer_servings + spirit_servings + wine_servings',
                         data = bebida_mundo)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

'''Na variável teste, será designado valores para uma nova previsao qualquer,
   atentando o fato de se colocar os valores na ordem das variáveis passadas anteriormente.
   Ex:beer_servings[0] = 400 copos
   spirit_servings[1] = 400 copos
   wine_servings[2] = 400 taças'''
 
teste = np.array([400, 400, 400])
teste = teste.reshape(1, -1)
modelo.predict(teste)

'''OBS³:  Nesse caso foi adotado que o volume de um copo de cerveja, destilado e vinho são de 300 ml (0.3 L), 100 ml (0.1 L)
e 450 ml (0.45 L), este que usará apenas 150 ml (0.15 L) por taça. Então supondo que uma pessoa consuma 400 copos/taças de cada
bebida, pode dizer que está consumindo 120 L, 40 L e 60 L de cerveja, destilados e vinho, respectivamente. No total, consumiu
cerca de 220 L de bebida alcoólica, dos quais, aproximadamente, 20.45 L são de puro álcool, correspondendo cerca de 9.3 % de 
álcool, aproximadamente.'''
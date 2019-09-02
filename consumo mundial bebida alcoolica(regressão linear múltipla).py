import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

''' OBS¹: Todos os 193 países foram selecionados, inclusive aqueles que são abstinentes, em todas as bebidas alcoólicas '''
'''OBS²: O consumo é per capita '''

bebida_mundo = pd.read_csv('D:\\Meus Documentos\Desktop\\Projetos Cientista de Dados\\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')

#Seleciona as colunas para serem as variáveis explanatória e resposta(X e y, respectivamente)
X = bebida_mundo.iloc[:, 1:4] #beer_servings, spirit_servings e wine_servings
y = bebida_mundo.iloc[:, 4] #total_litres_of_alcohol

modelo = LinearRegression()
modelo.fit(X, y)

# A função score serve para mostrar o grau de infuencia das variáveis explicativas sobre a variável de resposta.
modelo.score(X, y)

# As três linhas de comando a seguir tem a função de mostrar todos os valores detalhados, parecido no R
modelo_ajustado = sm.ols(formula = 'total_litres_of_pure_alcohol ~ beer_servings + spirit_servings + wine_servings',
                         data = bebida_mundo)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

'''Na variável teste, será designado valores para uma nova previsao qualquer,
 atentando o fato de se colocar os valores na ordem das variáveis passadas anteriormente.'''
 
teste = np.array([400, 400, 400])
'''Ex:beer_servings = 400 copos
   spirit_servings = 400 copos
   wine_servings = 400 taças '''
teste = teste.reshape(1, -1)
modelo.predict(teste)

'''OBS³:  Nesse caso foi adotado que o volume de um copo de cerveja, destilado e vinho são de 300 ml (0.3 L), 100 ml (0.1 L)
e 450 ml (0.45 L), este que usará apenas 150 ml (0.15 L) por taça. Então supondo que uma pessoa consuma 400 copos/taças de cada
bebida, pode dizer que está consumindo 120 L, 40 L e 60 L de cerveja, destilados e vinho, respectivamente. No total, consumiu
cerca de 220 L de bebida alcoólica, dos quais, aproximadamente, 20.67 L são de puro álcool, correspondendo cerca de 9.4 % de 
álcool, aproximadamente.'''

'''Obs: O matplotlib poderia ter sido usado, entretando com há mais de uma váriavel,
teriamos de importar um modelo que suporte mais de 2 dimensões, sem contar no poder
de processamento que seria necessário para gerar o gráfico. '''
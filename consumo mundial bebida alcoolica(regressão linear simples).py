import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

''' OBS¹: Todos os 193 países foram selecionados, inclusive aqueles que são abstinentes, em todas as bebidas alcoólicas '''
'''OBS²: O consumo é per capita '''

bebida_mundo = pd.read_csv('D:\\Meus Documentos\Desktop\\Projetos Cientista de Dados\\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')

'''1) Relação linear com o total de alcool ingerido (em Litros) com o total de cerveja ingerida (em porções) '''

#Seleciona as colunas para serem as variáveis explanatória e resposta(X e y, respectivamente)
X = bebida_mundo.iloc[:, 1].values #beer_servings
y = bebida_mundo.iloc[:, 4].values #total_litres_of_alcohol
correlacao = np.corrcoef(X, y)

#A função reshape é necessaria para transformar o X em matriz, para assim ser colocada no FIT
#Lembrando que com a transformaçao de X em matriz, só será possível executar a correlação no seu formato original
X = X.reshape(-1, 1)
modelo1 = LinearRegression()
modelo1.fit(X, y)

#Mostra a interceptação e inclinação do modelo(intercept e coef, respectivamente)
modelo1.intercept_
modelo1.coef_

#Plotar o gráfico com a linha de regressão é necessário executar os 4 comandos plt juntos
plt.scatter(X, y)
plt.xlabel('Copos de cerveja por pessoa (por ano)')
plt.ylabel('Total de litros de álcool ingeridos (por ano)')
plt.plot(X, modelo1.predict(X), color = 'red')

#Calculo manual com o modelo treinado para qualquer achar o y (x[beer_servings] = 400)
modelo1.intercept_ + modelo1.coef_ * 400

#Cálculo automático da máquina
modelo1.predict([[400]])

''' Como no estudo não informou uma porção em litros como referência, podemos fazer suposições a partir desse modelo
por exemplo, se uma pessoa bebe 400 copos de cerveja por ano, e adotando um copo de cerveja com 300 ml (0,3L), uma
pessoa que bebe 400 copos por ano (dependendo de cada país, obviamente), bebe 120 litros de cerveja, e só de alcool puro,
uma pessoa bebe, aproximadamente, 13.88 litros de álcool (cerca de 11.56% aproximadamente) '''

#Visualização dos resíduos(resultado entre a distância dos pontos com a linha de referência)
modelo1._residues

#Visualização dos resíduos no gráfico
visualizador1 = ResidualsPlot(modelo1)
visualizador1.fit(X, y)
visualizador1.poof()

#Os resíduos quando mais próximo de zero, melhor o modelo

'''2) Relação linear entre total de álcool ingerido (em Litros) com o total de destilados ingerido (em porções)
   OBS: Bebidas destiladas são todas que tiveram seu processo de destilação (vodca, uísque, tequila, rum, dentre outros)  '''

A = bebida_mundo.iloc[:, 2].values #spirit_servings 
b = bebida_mundo.iloc[:, 4].values #total_litres_of_alcohol
correlacao2 = np.corrcoef(A, b)

A = A.reshape(-1, 1)
modelo2 = LinearRegression()
modelo2.fit(A, b)

modelo2.intercept_
modelo2.coef_

plt.scatter(A, b)
plt.xlabel('Copos de destilados por pessoa (por ano)')
plt.ylabel('Total de litros de álcool ingeridos (por ano)')
plt.plot(A, modelo2.predict(A), color = 'yellow')

modelo2.intercept_ + modelo2.coef_ * 400

modelo2.predict([[400]])

''' Como no estudo não informou uma porção em litros como referência, podemos fazer suposições a partir desse modelo
por exemplo, se uma pessoa bebe 400 copos de destilados por ano, e adotando um copo de destilado com 100 ml (0,1L), uma
pessoa que bebe 400 copos por ano (dependendo de cada país, obviamente), bebe 40 litros de destilado, e só de alcool puro,
uma pessoa bebe, aproximadamente, 13.65 litros de álcool (cerca de 34.12% aproximadamente) '''

modelo2._residues

visualizador2 = ResidualsPlot(modelo2)
visualizador2.fit(A, b)
visualizador2.poof() 

'''3) Relação linear entre total de álcool ingerido (em Litros) em relação com o total de vinho ingerido (em porções) ''' 

C = bebida_mundo.iloc[:, 3].values #wine_servings 
d = bebida_mundo.iloc[:, 4].values #total_litres_of_alcohol
correlacao3 = np.corrcoef(C, d)

C = C.reshape(-1, 1)
modelo3 = LinearRegression()
modelo3.fit(C, d)

modelo3.intercept_
modelo3.coef_

plt.scatter(C, d)
plt.xlabel('Taças de vinho por pessoa (por ano)')
plt.ylabel('Total de litros de álcool ingeridos (por ano)')
plt.plot(C, modelo3.predict(C), color = 'purple')

modelo3.intercept_ + modelo3.coef_ * 400

modelo3.predict([[400]]) 

''' Como no estudo não informou uma porção em litros como referência, podemos fazer suposições a partir desse modelo
por exemplo, se uma pessoa bebe 400 copos de destilados por ano, e adotando uma taça de de vinho com 450 ml (0,45L),
entretanto uma pessoa só bebe cerca de 1/3 da taça então uma pessoa bebe 150 ml (0,15L), uma pessoa que bebe 400 copos
por ano (dependendo de cada país, obviamente), bebe 60 litros de vinho, e só de alcool puro,uma pessoa bebe,
aproximadamente, 15.8 litros de álcool (cerca de 26.3% aproximadamente) '''

modelo3._residues

visualizador3 = ResidualsPlot(modelo3)
visualizador3.fit(C, d)
visualizador3.poof() 

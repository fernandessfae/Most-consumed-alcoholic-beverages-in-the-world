import pandas as pd
from sklearn.metrics import confusion_matrix
import skfuzzy

''' Aqui vamos utilizar o fuzzy c-means(agrupamento parcial difuso), este que é utilizado para que elementos
    possam pertencer a mais de um grupo, com todos os países, exceto aqueles que não consomem bebida alcoólica.'''

#Carregamento da base de dados de todos os países, e remoção dos países não alcoólicos
bebida_mundo = pd.read_csv('Consumo de bebidas alcoólicas ao redor do mundo.csv')
for index, row in bebida_mundo.iterrows():
    if row['total_litres_of_pure_alcohol'] == 0:
        bebida_mundo = bebida_mundo.drop(index)

'''Atenção: nessa novas colunas atribuirei valores numéricos e nominais para cada categoria de bebida: 0 - beer, 1 - spirit, 2 - wine, 3 - none '''
nova_coluna_numerica = []
nova_coluna_nominal = []

for index, column in bebida_mundo.iterrows():
   if column['beer_servings'] > column['spirit_servings'] and column['beer_servings'] > column['wine_servings']:
       nova_coluna_numerica.append(0)
       nova_coluna_nominal.append('beer')
   elif column['spirit_servings'] > column['beer_servings'] and column['spirit_servings'] > column['wine_servings']:
       nova_coluna_numerica.append(1)
       nova_coluna_nominal.append('spirit')
   elif column['wine_servings'] > column['beer_servings'] and column['wine_servings'] > column['spirit_servings']:
       nova_coluna_numerica.append(2)
       nova_coluna_nominal.append('wine')

#Adicionando as novas colunas
bebida_mundo['most_consumed_number'] = nova_coluna_numerica
bebida_mundo['most_consumed_nominal'] = nova_coluna_nominal 

#Criação de uma variável com as colunas númericas da quantidade ingerida de cada classe
bebida = bebida_mundo.iloc[:, 1:4].values

#Preparação do fuzzy c-means
r = skfuzzy.cmeans(data = bebida.T, c = 3, m = 2, error = 0.005, maxiter = 1000, init = None)

#Variável com o agrupamento fuzzy
previsoes_porcentagem = r[1] 

previsoes_porcentagem[0][0] # Probabilidade de está no grupo 0
previsoes_porcentagem[1][0] # Probabilidade de está no grupo 1
previsoes_porcentagem[2][0] # Probabilidade de está no grupo 2

#Pega o maior valor de cada linha e define o grupo de um país com a bebida mais consumida
previsoes = previsoes_porcentagem.argmax(axis = 0)

resultados = confusion_matrix(bebida_mundo['most_consumed_number'], previsoes)

'''No caso dos agrupamentos Kmeans/Kmedoids/c-fuzzy means, NÃO se faz a leitura da diagonal principal, já que esse modelo é de agrupamento, e não de classificação previsão.
   Ou seja, os maiores valores de cada coluna representa cada grupo em si. Nesse caso coluna0 = beer, coluna1 = spirit e coluna2 = wine.'''
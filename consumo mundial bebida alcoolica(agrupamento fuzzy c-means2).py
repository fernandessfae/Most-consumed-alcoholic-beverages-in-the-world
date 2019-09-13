import pandas as pd
from sklearn.metrics import confusion_matrix
import skfuzzy

''' Aqui vamos utilizar o fuzzy c-means(agrupamento parcial difuso), este que é utilizado para que elementos possam pertencer a mais de um grupo, com todos os países, exceto aqueles que não consomem bebida alcoólica.'''

#Carregamento da base de dados
bebida_mundo = pd.read_csv('D:\Meus Documentos\Desktop\Projetos Cientista de Dados\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')
bebida_mundo = bebida_mundo.drop([0, 13, 46, 79, 90, 97, 103, 106, 107, 111, 128, 147, 158])

#Usaremos uma 2 listas(numerica e nominal), junto com um loop para criar uma nova coluna com os grupos definidos pelo laço
'''Atenção: nessa novas colunas atribuirei valores numéricos e nominais para cada categoria de bebida: 0 - beer, 1 - spirit, 2 - wine'''
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
##bebida = bebida.T 

#Preparação do fuzzy c-means
r = skfuzzy.cmeans(data = bebida.T, c = 3, m = 2, error = 0.005, maxiter = 1000, init = None)

#Para saber como funciona melhor cada parâmetro do skfuzzy.cmeans, basta acessar o link https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.html#cmeans

'''Obs¹: É necessário passar a matriz transposta para que o algoritmo funcione bem e tenha uma melhor performance
então pode fazer com que no parâmetro 'data' passe a variável.T(que significa transposta), ou já fazer a transformação
antes de passar a variável, que está representada com ##. '''

'''Obs²: Depois de executar a variável r, está gerará 7 linhas com valores, mas a que iremos utilizar terá o indice 1
e então faremos uma variável para ela. '''

previsoes_porcentagem = r[1] 

previsoes_porcentagem[0][0] # Probabilidade de está no grupo 0
previsoes_porcentagem[1][0] # Probabilidade de está no grupo 1
previsoes_porcentagem[2][0] # Probabilidade de está no grupo 2

previsoes = previsoes_porcentagem.argmax(axis = 0)

resultados = confusion_matrix(bebida_mundo['most_consumed_number'], previsoes)

'''No caso dos agrupamentos Kmeans/Kmedoids/c-fuzzy means, NÃO se faz a leitura da diagonal principal, já que esse modelo é de agrupamento, e não de classificação previsão.
   Ou seja, os maiores valores de cada coluna representa cada grupo em si. Nesse caso coluna0 = beer, coluna1 = spirit e coluna2 = wine.'''
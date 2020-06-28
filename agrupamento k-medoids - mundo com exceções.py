import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

''' Aqui iremos utilizar o agrupamento K-medoids (Algoritmo PAM), que é um algoritmo de agrupamento que lembra o algoritmo k-means, só que mais automatizado, e sem os países que não consomem bebida alcoólica.'''

#Carregamento da base de dados de todos os países, e remoção dos países não alcoólicos
bebida_mundo = pd.read_csv('Consumo de bebidas alcoólicas ao redor do mundo.csv')
for index, row in bebida_mundo.iterrows():
    if row['total_litres_of_pure_alcohol'] == 0:
        bebida_mundo = bebida_mundo.drop(index)

'''Atenção: nessa novas colunas atribuirei valores numéricos e nominais para cada categoria de bebida: 0 - beer, 1 - spirit, 2 - wine '''
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

#Criação de uma variável com as classes dos registros
bebida_numero = bebida_mundo.iloc[:, 5].values

#Faz o processo de achar o kmedoids automaticamente
cluster = kmedoids(bebida, [117, 68, 61])
cluster.get_medoids()

#Faz o processamento de clusterização
cluster.process()

#A variavel previsoes determina o número de cluster que a maquina ja processou anteriormente
previsoes = cluster.get_clusters()

#A variavel medoides determina o medoide (centro de um cluster)
medoides = cluster.get_medoids()

#Gera um gráfico com os 3 grupos, onde a * é o centro dos medoides
v = cluster_visualizer()
v.append_clusters(previsoes, bebida)
v.append_cluster(medoides, bebida, marker = '*', markersize = 100)
v.show()

#Criaremos 2 listas para separar os valores previstos pela máquina, e os valores reais de cada classe de cada elemento
lista_previsoes = []
lista_real = []

for i in range(len(previsoes)):
    for j in range(len(previsoes[i])):
        lista_previsoes.append(i)
        lista_real.append(bebida_numero[previsoes[i][j]])   

#Transfoma ambas as listas num array
lista_previsoes = np.asarray(lista_previsoes)
lista_real = np.asarray(lista_real)

#Gera uma matriz de confusão
resultados = confusion_matrix(lista_real, lista_previsoes)

'''No caso dos agrupamentos Kmeans/Kmedoids/c-fuzzy means, NÃO se faz a leitura da diagonal principal, já que esse modelo é de agrupamento, e não de classificação previsão.
   Ou seja, os maiores valores de cada coluna representa cada grupo em si. Nesse caso coluna0 = beer, coluna1 = spirit e coluna2 = wine. E percebemos que dos 180 países,
   o modelo conseguiu agrupar 127 países corretamente, representando um acerto de 70.5%, aproximadamente.'''
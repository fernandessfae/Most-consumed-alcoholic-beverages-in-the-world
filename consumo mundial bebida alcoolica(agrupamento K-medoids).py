import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

''' Aqui iremos utilizar o agrupamento K-medoids (Algoritmo PAM), que é um algoritmo de agrupamento que lembra o algoritmo k-means, só que mais automatizado.'''

#Carregamento da base de dados
bebida_mundo = pd.read_csv('D:\Meus Documentos\Desktop\Projetos Cientista de Dados\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')

#Usaremos uma 2 listas(numerica e nominal), junto com um loop para criar uma nova coluna com os grupos definidos pelo laço
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
   elif column['wine_servings'] == column['beer_servings'] and column['wine_servings'] == column['spirit_servings']:
       nova_coluna_numerica.append(3)
       nova_coluna_nominal.append('none')
   
#Adicionando as novas colunas
bebida_mundo['most_consumed_number'] = nova_coluna_numerica
bebida_mundo['most_consumed_nominal'] = nova_coluna_nominal 

#Criação de uma variável com as colunas númericas da quantidade ingerida de cada classe
bebida = bebida_mundo.iloc[:, 1:4].values

#Criação de uma variável com as classes dos registros
bebida_numero = bebida_mundo.iloc[:, 5].values

#Faz o processo de achar o kmedoids autmoaticamente (executar os 2 comandos simultaneamente)
cluster = kmedoids(bebida, [117, 68, 61])
cluster.get_medoids()

#Faz o processamento de clusterização
cluster.process()

#A variavel previsoes determina o número de cluster que a maquina ja processou anteriormente com o clusters.process()
previsoes = cluster.get_clusters()

#A variavel medoides determina o medoide (centro de um cluster)
medoides = cluster.get_medoids()

#Gera um gráfico com os 3 grupos, onde a * é o centro dos medoides (executar os 4 comandos simultâneos)
v = cluster_visualizer()
v.append_clusters(previsoes, bebida)
v.append_cluster(medoides, bebida, marker = '*', markersize = 100)
v.show()

#Criaremos 2 listas para separar os valores previstos pela máquina, e os valores reais de cada classe de cada elemento
lista_previsoes = []
lista_real = []

#E aqui está a estrutura de repetição onde irá segregar cada elemento em sua lista
for i in range(len(previsoes)):
    for j in range(len(previsoes[i])):
        lista_previsoes.append(i)
        lista_real.append(bebida_numero[previsoes[i][j]])

''' Para entendermos como funciona esse laço, faremos tudo passo a passo:
    1) No (for i in range(len(previsoes)), ele irá contar quantos indices a variável previsoes tem, que no caso possuem 3 indices(com 3 listas,cada),
       onde o indice representa a classe de cada elemento e a lista representam os elementos
    2) No (for j in range(len(previsoes[i])), ele irá contar quantos elementos existem nos 3 indicies (ou seja, irá percorrer cada elemento
       da lista
    3) Na lista_previsoes.append(i) a máquina pega o valor que foi predito e coloca nessa lista
    4) Na lista_real.append(bebida_numero[previsoes[i][j]]) a maquina pega o valor real da classe daquele elemento e coloca nessa lista
       '''    

#Transfoma ambas as listas num array
lista_previsoes = np.asarray(lista_previsoes)
lista_real = np.asarray(lista_real)

#Gera uma matriz de confusão
resultados = confusion_matrix(lista_real, lista_previsoes)

'''No caso dos agrupamentos Kmeans/Kmedoids/c-fuzzy means, NÃO se faz a leitura da diagonal principal, já que esse modelo é de agrupamento, e não de classificação previsão.
   Ou seja, os maiores valores de cada coluna representa cada grupo em si. Nesse caso coluna0 = beer, coluna1 = spirit, coluna2 = wine e coluna3 = none. E percebemos que dos 193 países,
   o modelo conseguiu agrupar 109 países corretamente, representando um acerto de 56.5%, aproximadamente.'''    
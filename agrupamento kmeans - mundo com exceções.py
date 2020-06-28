import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

''' Aqui vamos utilizar o Kmeans, este que é utilizado para que elementos possam pertencer ao grupo mais próximo da média. Usaremos em todos os países, excetos aqueles que não consomem bebida alcoólica.'''

#Carregamento da base de dados de todos os países, e remoção dos países não alcoólicos
bebida_mundo = pd.read_csv('Consumo de bebidas alcoólicas ao redor do mundo.csv')
for index, row in bebida_mundo.iterrows():
    if row['total_litres_of_pure_alcohol'] == 0:
        bebida_mundo = bebida_mundo.drop(index)

'''Atenção: nessa nova coluna atribuirei valores numéricos/nominais para cada categoria de país, de acordo com
   a sua bebida marjoritária: 0 - beer, 1 - spirit, 2 - wine'''
   
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

#Criaremos uma variável de como cada pais consome as bebidas alcoólicas
bebida = bebida_mundo.iloc[:, 1:4].values

#Criação de uma variável para a contagem de elementos únicos e o total de elementos de cada classe
unicos, quantidade = np.unique(bebida_mundo['most_consumed_number'], return_counts = True)

#Definação do número de clusters
cluster = KMeans(n_clusters = 3)
cluster.fit(bebida)

#Definição dos centróides e previsões as amostras
centroides = cluster.cluster_centers_
previsoes = cluster.labels_

#Criação de uma variável para a contagem de elementos de cada classe com os valores previstos
unicos2, quantidade2 = np.unique(previsoes, return_counts = True)

#Criaremos variável com uma matriz de confusão, passando como parâmetro os valores corretos e aqueles previstos pela maquina
resultados = confusion_matrix(bebida_mundo['most_consumed_number'], previsoes)

'''No caso dos agrupamentos Kmeans/Kmedoids/c-fuzzy means, NÃO se faz a leitura da diagonal principal, já que esse modelo é de agrupamento, e não de classificação previsão.
   Ou seja, os maiores valores de cada coluna representa cada grupo em si. Nesse caso coluna0 = beer, coluna1 = spirit e coluna2 = wine. E percebemos que dos 180 países,
   o modelo conseguiu agrupar 126 países corretamente, representando um acerto de 70%.'''

#Plotando o gráfico do agrupamento KMeans, executando os 4 comandos ao mesmo tempo
plt.figure(figsize = (10, 5))
plt.title('Agrupamento dos países com as suas bebidas marjoritárias', fontsize = 16)
plt.scatter(bebida[previsoes == 0, 0], bebida[previsoes == 0, 1], c = 'green', label = 'beer')
plt.scatter(bebida[previsoes == 1, 1], bebida[previsoes == 1, 0], c = 'red', label = 'spirit')
plt.scatter(bebida[previsoes == 2, 2], bebida[previsoes == 2, 0], c = 'blue', label = 'wine')
plt.legend()

'''Exemplificando: bebida[previsoes == 0(0 - beer), 0(coluna beer_servings)], bebida[previsoes == 0(0 - beer), 1(spirit_servings)]''' 
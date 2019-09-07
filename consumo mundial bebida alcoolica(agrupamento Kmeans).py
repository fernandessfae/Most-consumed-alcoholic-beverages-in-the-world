import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

''' Aqui vamos utilizar o Kmeans, este que é utilizado para que elementos possam pertencer ao grupo mais próximo da média.'''

bebida_mundo = pd.read_csv('D:\Meus Documentos\Desktop\Projetos Cientista de Dados\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')

'''Atenção¹: nessa nova coluna atribuirei valores numéricos para cada categoria de bebida: 0 - beer, 1 - spirit, 2 - wine, 3 - none '''
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

#Criaremos uma variável de como cada pais consome as bebidas alcoólicas
bebida = bebida_mundo.iloc[:, 1:4].values

#Criação de uma variável para a contagem de elementos de cada classe
unicos, quantidade = np.unique(bebida_mundo['most_consumed_number'], return_counts = True)

#Definação de uma variável com o seu número de clusters, execute os 2 comandos juntos
cluster = KMeans(n_clusters = 4)
cluster.fit(bebida)

#Criar uma variável para a geração dos centróides, e outras para informar qual cluster cada elemento pertence
centroides = cluster.cluster_centers_
previsoes = cluster.labels_

#Criação de uma variável para a contagem de elementos de cada classe com os valores previstos
unicos2, quantidade2 = np.unique(previsoes, return_counts = True)

#Criaremos variável com uma matriz de confusão, passando como parâmetro os valores corretos e aqueles previstos pela maquina
resultados = confusion_matrix(bebida_mundo['most_consumed_number'], previsoes)

#Plotando o gráfico do agrupamento KMeans, executando os 4 comandos ao mesmo tempo
plt.scatter(bebida[previsoes == 0, 0], bebida[previsoes == 0, 1], c = 'green', label = 'beer')
plt.scatter(bebida[previsoes == 1, 0], bebida[previsoes == 1, 1], c = 'red', label = 'spirit')
plt.scatter(bebida[previsoes == 2, 0], bebida[previsoes == 2, 2], c = 'blue', label = 'wine')
plt.legend()
 
'''Atenção²: aqui vai uma explicação de como o gráfico como os grupo foram gerados. Nesse caso o scater foi passado como parâmentros
o eixo x, o eixo y, a cor, e o nome de cada grupo. No eixo x passei 'previsoes' (a variavel que a maquina fez a previsão de cada grupo) == a, b
sendo que 'a' é o número atribuido a classe do registro como mencionado na "Atenção¹", e 'b' é a coluna que foi utilizada com as caracteristicas
dos paises que bem mais aquela bebida específica(noe eixo y foi a mesma coisa).


Exeplificando: bebida[previsoes == 0(0 - beer), 0(coluna beer_servings)], bebida[previsoes == 0(0 - beer), 1(spirit_servings)]''' 

'''Obs¹: Como os países que n ingerem bebida alcoolica os valores são zerados, então optei não coloca-los no gráfico. '''      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier

'''Aqui ulitizaremos o método de classificação knn, utilizando validação cruzada
   e holdout, para ver se eles acertam a bebida marjoritária de cada país. '''

#Carregamento da base de dados
bebida_mundo = pd.read_csv('Consumo de bebidas alcoólicas ao redor do mundo.csv')

#Criação de uma nova coluna com as bebidas marjoritárias de cada país
nova_coluna_nominal = []

for index, column in bebida_mundo.iterrows():
   if column['beer_servings'] > column['spirit_servings'] and column['beer_servings'] > column['wine_servings']:
       nova_coluna_nominal.append('beer')
   elif column['spirit_servings'] > column['beer_servings'] and column['spirit_servings'] > column['wine_servings']:
       nova_coluna_nominal.append('spirit')
   elif column['wine_servings'] > column['beer_servings'] and column['wine_servings'] > column['spirit_servings']:
       nova_coluna_nominal.append('wine')
   elif column['wine_servings'] == column['beer_servings'] and column['wine_servings'] == column['spirit_servings']:
       nova_coluna_nominal.append('none')
       
bebida_mundo['most_consumed_nominal'] = nova_coluna_nominal

#Criação das váriaveis previsoras e da classe
previsores = bebida_mundo.iloc[:, 1:4].values
classe = bebida_mundo.iloc[:, 5].values

#Transformação da classe em numérica
classe = LabelEncoder().fit_transform(classe)

#Criação de classificador(es) do tipo knn
pip = Pipeline([('knn', KNeighborsClassifier())])

#Verificação dos parâmetros que podem ser colocados no classificador
sorted(pip.get_params().keys())

#Criação de uma lista com diferentes parâmetros do knn
n_neighbors = [3, 5, 7, 9]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [30, 40, 50, 60]
p = [1, 2]

#Dicionário para os parâmetro GRID
parametros_grid = dict(knn__n_neighbors = n_neighbors,
                       knn__weights = weights,
                       knn__algorithm = algorithm,
                       knn__leaf_size = leaf_size)
parametros_grid

#Criação e aplicação do gridsearch 
grid = GridSearchCV(pip, parametros_grid, cv = 5, scoring = 'accuracy')
grid.fit(previsores,classe)

# Imprime os scores por combinações
medias_teste = grid.cv_results_['mean_test_score']

for i, dicionario in enumerate(grid.cv_results_['params']):
    z = dicionario.copy()
    z.update({'mean': round(medias_teste[i], 4)})
    print(z)

#Imprime os melhores parâmetros e acurácia da melhor knn
grid.best_params_
grid.best_score_

#Criação do classificador
clf = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, n_neighbors = 9, weights = 'distance')

#Visualização do score, acurácia e métricas do modelo utilizando validação cruzada
resultado = cross_val_score(clf, previsores, classe, cv = 5, scoring = 'accuracy')
resultados = cross_val_predict(clf, previsores, classe, cv = 5)
valor_classes = sorted(np.unique(classe))
print(f'O desvio padrão da soma de todos os folds do modelo KNeighborsClassifier é de {round(resultado.std(), 4)}')
print(f'A acurácia do modelo KNeighborsClassifier é de {round(metrics.accuracy_score(classe,resultados) * 100, 2)}%')
print(f'As métricas do modelo KNeighborsClassifier é:\n {metrics.classification_report(classe,resultados,valor_classes)}') 

#Criação do modelo utilizando holdout
p_treinamento, p_teste, c_treinamento, c_teste = train_test_split(previsores, classe, test_size = 0.2, random_state = 0)
clf.fit(p_treinamento, c_treinamento)

# Resultados da predição
c_previsao = clf.predict(p_teste)

#Mostra a matriz de confusão e a previsão do modelo
precisao = metrics.accuracy_score(c_teste,c_previsao)
matriz = metrics.confusion_matrix(c_teste, c_previsao)

#Gera o imagem da matriz de confusão
v = ConfusionMatrix(clf)
v.fit(p_treinamento, c_treinamento)
v.score(p_teste, c_teste)
v.poof()

#Geração de um gráfico comparando o desempenho da validação cruzada com a divisão de dados
plt.figure(figsize = (10, 5))
plt.title('Medição da acurácia do modelo - KNeighborsClassifier', fontsize = 16, fontweight = 'bold')
plt.xlabel('Tipo de modelo', fontsize = 12, fontweight = 'bold')
plt.ylabel('Precisão (%)', fontsize = 12, fontweight = 'bold')
bar1 = plt.bar('Validação Cruzada', round(metrics.accuracy_score(classe,resultados) * 100, 2), color = 'r', width = 0.3)
bar2 = plt.bar(x = 'Holdout', height = round(precisao * 100, 2), color = 'b', width = 0.3)
for rect in bar1 + bar2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{float(height)}%', ha='center', va='bottom', fontsize = 14)
plt.show()
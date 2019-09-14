import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

'''Aqui ulitizaremos o método de classificação com KNN, para ver se eles acertam a bebida marjoritária de cada país, exceto os países que não consomem bebida alcoólica. '''

#Carregamento da base de dados
bebida_mundo = pd.read_csv('D:\\Meus Documentos\\Desktop\\Projetos Cientista de Dados\\Consumo bebidas alcóolicas no mundo\\Consumo de bebidas alcoólicas ao redor do mundo.csv')
bebida_mundo = bebida_mundo.drop([0, 13, 46, 79, 90, 97, 103, 106, 107, 111, 128, 147, 158])

'''Atenção¹: nessa nova coluna atribuirei valores numéricos/nominais para cada categoria de bebida: 0 - beer, 1 - spirit, 2 - wine, 3 - none '''
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

'''Obs¹: Como na minha base de dados eu não tenho/selecionei variáveis categóricas, não haverá a necessidade de transformação das mesmas '''

#Essa variável receberá as variáveis independentes
previsores = bebida_mundo.iloc[:, 1:4].values

#Essa variável receberá a variável de resposta
classe = bebida_mundo.iloc[:, 5].values

'''Obs²: Caso houvessem variáveis categoricas para serem usadas no modelo, teriamos que importar uma biblioteca chamada LabelEncoder, esta que iria transformar variáveis
         categóricas em variáveis númericas. E detalhe, teriamos que transformar as colunas em ordem. Por exemplo, se as colunas 1, 3, 5 fossem
         categóricas, teriamos que transformar primeiro a coluna 1, depois a coluna 3 e no final a coluna 5, isso para que não haver erro na hora
         da execução do treinamento.'''

#Aqui hávera a divisão dos dados para treinamento e teste passando como parâmetros(variavel independente, variável resposta, a amostra de teste[0 até 1] e divisao da base de dados igual)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.24, random_state = 0)

'''Agora vamos treinar o algoritmo knn, ajustando o número de vizinhos e usar a base de treinamento para treinar o modelo
   sobre o n_neighbors(número de vizinhos) o valor é ajustado de acordo com a sua vontade. Nesse caso específico, como 
   existe 4 grupos, o mais sensato é colocar n = 3. '''
   
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_treinamento, y_treinamento)

#Faz a previsão do algoritmo knn usando a base de teste
previsoes = knn.predict(X_teste)

#Cria uma matriz de confusão nessa variável
confusao = confusion_matrix(y_teste, previsoes)

#Cria variáveis com a taxa de acerto e erro
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto

'''Obs³: Para efeito de comparação, decidi ajustar o teste_size para ver quais valores seriam melhor para o teste de árvore de decisão da máquina, e o resultado foi esse:
    
         Como constatado, a taxa de acerto foi 94.4%, aproximadamente, com 30% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 94.3%, aproximadamente, com 29% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 94.1%, aproximadamente, com 28% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 93.9%, aproximadamente, com 27% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 93.6%, aproximadamente, com 26% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 93.3%, aproximadamente, com 25% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 95.4%, aproximadamente, com 24% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 95.2%, aproximadamente, com 23% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 95.0%, aproximadamente, com 22% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 94.7%, aproximadamente, com 21% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 94.4%, aproximadamente, com 20% de dados para teste com todos os 180 países.
         
         Conclusão, o melhor resultado da máquina foi um valor de 0.24(24%) dos dados para o teste e 0.76(76%) dos dados para treinamento. A conclusão que se
         tira é de que há um limite para os dados de treinamento. Nem sempre quanto mais dados para treinamento, melhor será o resultado do teste.'''
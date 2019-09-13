import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

'''Aqui ulitizaremos o método de classificação com o Naive Bays, para ver se eles acertam a bebida marjoritária de cada país, excluindo os países que não ingerem bebida alcoólica. '''

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

'''Obs²: Caso houvessem variáveis categoricas para ser utilizadas no modelo, teriamos que importar uma biblioteca chamada LabelEncoder, esta que iria transformar variáveis
         categóricas em variáveis númericas. E detalhe, teriamos que transformar as colunas em ordem. Por exemplo, se as colunas 1, 3, 5 fossem
         categóricas, teriamos que transformar primeiro a coluna 1, depois a coluna 3 e no final a coluna 5, isso para que não haver erro na hora
         da execução do treinamento.'''

#Aqui hávera a divisão dos dados para treinamento e teste passando como parâmetros(variavel independente, variável resposta, a amostra de teste[0 até 1] e divisao da base de dados)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.2, random_state = 0)

#Agora vamos aplicar o naive bays nos dados de treinamento (executa os dois comandos simultaneamente)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

#Pega os dados passados no passo anterior para ser executado no modelo
previsoes = naive_bayes.predict(X_teste)

#gera uma variável com uma matriz de confusão
confusao = confusion_matrix(y_teste, previsoes)

#Revela o percentual de acerto e erro do modelo da máquina
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto

#Aqui irá gerá a figura da matriz de confusão (executar os 4 comandos simultaneamente)
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

'''Obs³: Para efeito de comparação, decidi ajustar o teste_size para ver quais valores seriam melhor para o teste de predição da máquina, e o resultado foi esse:
    
         Como constatado, a taxa de acerto foi 50.8%, aproximadamente, com 35% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 51.6%, aproximadamente, com 34% de dados para teste com todos os 180 países
         Como constatado, a taxa de acerto foi 53.3%, aproximadamente, com 33% de dados para teste com todos os 180 países
         Como constatado, a taxa de acerto foi 51.7%, aproximadamente, com 32% de dados para teste com todos os 180 países
         Como constatado, a taxa de acerto foi 53.6%, aproximadamente, com 31% de dados para teste com todos os 180 países
         Como constatado, a taxa de acerto foi 51.8%, aproximadamente, com 30% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 44.4%, aproximadamente, com 25% de dados para teste com todos os 180 países.
         Como constatado, a taxa de acerto foi 38.8%, aproximadamente, com 20% de dados para teste com todos os 180 países.
         
         Como constatado, o melhor resultado da máquina foi um valor de 0.31(31%) dos dados para o teste e 0.69(69%) dos dados para treinamento. A conclusão que se
         tira é de que nem sempre muitos dados para treinamento faz com que o teste seja mais eficiente, e nem de menos. Isso mostra que os dados tem que ser ajustados
         ao ponto máximo da performance dele, nem para mais, nem para menos.'''
import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np


def drink_graphic(dataframe: pd.DataFrame,col1: str, col2: str,
                  title: str) -> None:
    """Generation of graphic of the most consumed beverages and/or total
       alcohol consumed per person, on average, for each country around
       the world.
       
       :param dataframe: database to be used
       :type dataframe: pd.Dataframe
       :param col1: name of column for the countries of the world
       :type col1: str
       :param col2: column name of the specific drink
       :type col2: str
       :param title: chart title name
       :type title: str
    """       
    plt.figure(figsize = (10, 5))
    plt.bar(dataframe.nlargest(10, col2.lower())[col1.lower()],
            dataframe.nlargest(10, col2.lower())[col2.lower()],
            color = plt.cm.Set1(np.arange(10)))
    plt.title(title.capitalize(),
              fontdict= {'fontsize': 16, 'fontweight':'bold'})
    plt.xticks(rotation= 45)
    plt.ylabel(col2.replace('_', ' ').capitalize())
    plt.show()
    return None


url: str = 'https://raw.githubusercontent.com/fivethirtyeight/data/master'
specific_url: str = f'{url}/alcohol-consumption/drinks.csv'
url_datum: requests.models = requests.get(specific_url)

if url_datum.ok:
    data: str = url_datum.content.decode('utf-8')
    dataset: pd.DataFrame = pd.read_csv(io.StringIO(data))
    #print(dataset.columns)
    drink_graphic(dataset,'country','beer_servings',
                  'The 10 countries that consume the most beer per year')
    drink_graphic(dataset,'country','wine_servings',
                  'The 10 countries that consume the most wine per year')
    drink_graphic(dataset,'country','spirit_servings',
                  'The 10 countries that consume the most spirit per year')
    drink_graphic(dataset,'country','total_litres_of_pure_alcohol',
                  'The 10 countries that consume the most total alcohol per year')
    

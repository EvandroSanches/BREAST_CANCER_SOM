import numpy as np
import pandas as pd
import matplotlib.pylab as plb
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

#Linhas e Colunas do mapa = 5√n

#Carregando dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
classe = np.asarray(classe)

#Normalizando dados
normalizador = MinMaxScaler()
previsores = normalizador.fit_transform(previsores)

#Criando modelo
som = MiniSom(x=28, y=28, sigma=4.0, learning_rate=1.0, random_seed=8, input_len=30)

#Iniciando pesos
som.random_weights_init(previsores)

#Treinando modelo
som.train_random(data=previsores, num_iteration=700)

#Definindo marcadores
markers = ['s', 'o']
color = ['blue', 'red']

#Plotando mapa
plb.pcolor(som.distance_map().T)
plb.colorbar()

for i, x in enumerate(previsores):
    w = som.winner(x)

    plb.plot(w[0] + 0.5, w[1] + 0.5, markers[int(classe[i])], markerfacecolor='None',
             markersize=10, markeredgecolor=color[int(classe[i])], markeredgewidth=2)

plb.show()



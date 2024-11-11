# inicio dle trabajo de optimizacion
import numpy as np
import random
import math
import matplotlib.pyplot as plt

#Iniciare con una lista pequeña.
listaClientes=[(2, 3), (5, 4), (7, 2), (8, 8), (1, 5), (6, 6)]  
demandas=[10, 20, 30, 25, 15, 10]
deposito=(0,0)
capacidadMaxima=50
clientesMaximos=6

def calcular_Angulo(cliente):
    x = cliente
    y = cliente
    return math.atan2(y - deposito[1], x - deposito[0])

def heuristica_Barrido(listaClientes, demandas, capacidadMaxima):
    angulos= [(cliente, demanda, calcular_Angulo(cliente)) 
              for  cliente, demanda in zip(listaClientes, demandas)]
    angulos.sort(key = lambda x: x[2]) #ordenar por angulo

    clusters = []
    cluster_Actual=[]
    carga_Actual = 0

    for cliente, demanda, angulos in angulos:
        if carga_Actual + demanda <= capacidadMaxima:
            cluster_Actual.append(cliente)
            carga_Actual += demanda
        else:
            clusters.append(cluster_Actual)
            cluster_Actual = [cliente]
            carga_Actual = demanda
    if cluster_Actual:
        clusters.append(cluster_Actual)
    
    return clusters

#Realiza la heuristica de barrido
clusters = heuristica_Barrido(listaClientes, demandas, capacidadMaxima)
print("Clusters generados:", clusters)

# Funciones del algoritmo genetico

#Distancia toal de una ruta
def calculo_Total_Distancia(ruta):
    distancia_Total = 0
    punto_Actual = deposito
    for cliente in ruta:
        #Distancia euclediana calculando la norma del vector
        distancia_Total += np.linalg.norm(np.array(punto_Actual) - np.array(cliente))
        punto_Actual = cliente
    distancia_Total += np.linalg.norm(np.array(punto_Actual) - np.array(deposito))
    
    return distancia_Total

# Generador inicial de la poblacion, utilizare las siglas PI para hacer entender que es PoblacionInicial
def generador_de_PI(clusters, tamaño_Poblacion=10):
    pobladores = []

    for i in range(tamaño_Poblacion):
        individuo = [random.sample(cluster, len(cluster)) 
                     for cluster in clusters]
        pobladores.append(individuo)
    
    return pobladores

#Metodo de seleccion: Metodo de la ruleta
def seleccion_Ruleta(poblacion, puntuaciones):
    fitness_Total = sum( 1 / puntuacion for puntuacion in puntuaciones)
    eleccion = random.uniform(0, fitness_Total)
    actual = 0

    for individuo, puntuacion in zip(poblacion, puntuaciones):
        actual += 1 / puntuacion
        
        if actual > eleccion:
            return individuo
        
# Crossover parametrizado uniforme con reparacion




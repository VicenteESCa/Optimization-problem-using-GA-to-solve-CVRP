# inicio dle trabajo de optimizacion
import numpy as np
import random
import math
import matplotlib.pyplot as plt

""" Si se desea no mostrar graficamente poner las comillas en el final de esta funcion. Las comillas:-> """
#Funcion para graficar el mambo: 
def graficar_ruta(ruta, generacion):
    plt.figure(figsize=(10, 6))

    # Graficamos las rutas de los clientes
    for cluster in ruta:
        x = [cliente[0] for cliente in cluster]
        y = [cliente[1] for cliente in cluster]
        
        # Generamos un color único para cada clúster
        color = np.random.rand(3,)  # Un color aleatorio para cada clúster
        
        # Graficamos la ruta del clúster (ida)
        plt.plot(x, y, marker='o', linestyle='--', color=color, markersize=5)
        
        # Agregar la flecha de ida (del depósito al primer cliente)
        plt.annotate('', xy=(x[0], y[0]), xytext=(deposito[0], deposito[1]),
                     arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="->", lw=1.5))
        
        # Agregar la flecha de vuelta (del último cliente al depósito)
        plt.annotate('', xy=(deposito[0], deposito[1]), xytext=(x[-1], y[-1]),
                     arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="->", lw=1.5))

    # Graficamos el depósito
    deposito_x, deposito_y = deposito
    plt.plot(deposito_x, deposito_y, marker='D', color='red', markersize=10)

    # Título y etiquetas
    plt.title(f"Generación {generacion + 1} - Mejor Ruta")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.show()


#Leer estructura del archivo:
def leer_archivo_vrp(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
    
    capacidad = None
    deposito = None
    listaClientes = []
    demandas = []
    
    seccion = None
    
    for linea in lineas:
        linea = linea.strip()
        
        if linea.startswith("CAPACITY"):
            capacidad = int(linea.split(":")[1].strip())
        elif linea == "NODE_COORD_SECTION":
            seccion = "NODE_COORD_SECTION"
        elif linea == "DEMAND_SECTION":
            seccion = "DEMAND_SECTION"
        elif linea == "DEPOT_SECTION":
            seccion = "DEPOT_SECTION"
        elif linea == "EOF":
            break
        elif seccion == "NODE_COORD_SECTION":
            partes = linea.split()
            id_cliente = int(partes[0])
            x = int(partes[1])
            y = int(partes[2])
            listaClientes.append((x, y))
        elif seccion == "DEMAND_SECTION":
            partes = linea.split()
            id_cliente = int(partes[0])
            demanda = int(partes[1])
            demandas.append(demanda)
        elif seccion == "DEPOT_SECTION":
            if int(linea) != -1:
                deposito = listaClientes[int(linea) - 1]

    return capacidad, deposito, listaClientes, demandas

#Declaracion de variables
capacidadMaxima, deposito, listaClientes, demandas = leer_archivo_vrp('./Dificil.vrp')

# Ejemplo de uso
def calcular_Angulo(cliente):
    x, y = cliente
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
def crossover_Parametrizado_Uniforme(padre, madre, probabilidad = 0.5):
    hijo = [None] * len(padre)
    genes_Usados = set()

    # Adicion del material genetico
    for i in range(len(padre)):
        if random.random() < probabilidad and padre[i] not in genes_Usados:
            hijo[i] = padre[i]
            genes_Usados.add(padre[i])
        elif madre[i] not in genes_Usados:
            hijo[i] = madre[i]
            genes_Usados.add(madre[i])

    # Completacion de genes faltantes
    for i in range(len(hijo)):
        if hijo[i] is None:
            for gen in padre:
                if gen not in genes_Usados:
                    hijo[i] = gen
                    genes_Usados.add(gen)
                    break
    
    return hijo

# Mutacion controlada
def mutar(individuo, prob_Mutar = 0.02):
    for cluster in individuo:
        if len(cluster)>1 and random.random()<prob_Mutar:
            i, j = random.sample(range(len(cluster)), 2)
            cluster[i], cluster[j] = cluster[j], cluster[i]   

# Algoritmo genetico de Chu-Beasly 
def algoritmo_Genetico(clusters, tamaño_Poblacional, maximo_sin_mejora, determina_boole, generaciones = 100, prob_Mutar=0.01):
    
    poblacion = generador_de_PI(clusters, tamaño_Poblacional)
    mejor_Puntuacion_Global = float('inf') #Metodo para definir como infinito positivoc d
    generaciones_Sin_Mejora = 0
    mejor_solucion_Global = None
    numero_De_Generacion = 0

    for generacion in range(generaciones):

        puntuaciones = [sum(calculo_Total_Distancia(cluster) for cluster in individuo) for individuo in poblacion]
        mejor_Puntuacion_Actual = min(puntuaciones)

        # Encontrar la mejor ruta de esta generación
        mejor_ruta_generacion = min(poblacion, key=lambda ind: sum(calculo_Total_Distancia(cluster) for cluster in ind))
        
        print(f"Generación {generacion + 1} - Mejor puntuación: {mejor_Puntuacion_Actual}")
        print(f"Mejor ruta en esta generación:")

        #Check de mehora significativa

        if mejor_Puntuacion_Actual < mejor_Puntuacion_Global:
            mejor_Puntuacion_Global = mejor_Puntuacion_Actual
            mejor_solucion_Global = mejor_ruta_generacion
            numero_De_Generacion = generacion
            print(f"Nuevo óptimo global encontrado: {mejor_Puntuacion_Global}")
            if determina_boole == True:
                graficar_ruta(mejor_solucion_Global, numero_De_Generacion)
                generaciones_Sin_Mejora = 0
            else:
                generaciones_Sin_Mejora = 0
        else:
            generaciones_Sin_Mejora += 1
        
        if generaciones_Sin_Mejora >= maximo_sin_mejora:
            break

        #Crear nueva generacion
        generacion_Siguiente = []
        for _ in range(tamaño_Poblacional // 2):
            padre = seleccion_Ruleta(poblacion, puntuaciones)
            madre = seleccion_Ruleta(poblacion, puntuaciones)
            hijo1 = [crossover_Parametrizado_Uniforme(cluster1, cluster2) for cluster1, cluster2 in zip(padre,madre)]
            hijo2 = [crossover_Parametrizado_Uniforme(cluster2, cluster1) for cluster1, cluster2 in zip(padre,madre)]
            mutar(hijo1, prob_Mutar)
            mutar(hijo2, prob_Mutar)
            generacion_Siguiente.extend([hijo1, hijo2])
        

        poblacion = generacion_Siguiente

    if mejor_Puntuacion_Global:
        graficar_ruta(mejor_ruta_generacion, generacion) # <- Esta linea

    return mejor_solucion_Global, mejor_Puntuacion_Global, numero_De_Generacion



tam_Poblacional = int(input("Defina el tamaño poblacional de las generaciones: "))
max_mejora = int(input(f"Defina el maximo de generaciones sin tener una mejora: "))
ver_Optimos = input("Determine con Y / N si desea ver los optimos cada que se encuentra uno nuevo: ")
condicion_del_booleano = None
if ver_Optimos == "Y":
    condicion_del_booleano = True
else:
    condicion_del_booleano = False
mejor_Ruta , mejor_Puntuacion, Numero_De_Generacion = algoritmo_Genetico(clusters, tam_Poblacional, max_mejora, condicion_del_booleano)
print(f"Mejor ruta encontrada {mejor_Ruta}")
print(f"Con una puntuacion de: {mejor_Puntuacion}")
graficar_ruta(mejor_Ruta, Numero_De_Generacion)

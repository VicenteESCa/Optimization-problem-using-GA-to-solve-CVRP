import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy

# --- 1. LECTURA Y PREPARACIÓN DE DATOS (Igual que antes) ---
def leer_archivo_vrp(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
    capacidad = None; deposito = None; listaClientes = []; demandas = []
    seccion = None
    for linea in lineas:
        linea = linea.strip()
        if linea.startswith("CAPACITY"): capacidad = int(linea.split(":")[1].strip())
        elif linea == "NODE_COORD_SECTION": seccion = "NODE_COORD_SECTION"
        elif linea == "DEMAND_SECTION": seccion = "DEMAND_SECTION"
        elif linea == "DEPOT_SECTION": seccion = "DEPOT_SECTION"
        elif linea == "EOF": break
        elif seccion == "NODE_COORD_SECTION":
            partes = linea.split()
            listaClientes.append((int(partes[1]), int(partes[2])))
        elif seccion == "DEMAND_SECTION":
            partes = linea.split()
            demandas.append(int(partes[1]))
        elif seccion == "DEPOT_SECTION":
            if int(linea) != -1: deposito = listaClientes[int(linea) - 1]
    return capacidad, deposito, listaClientes, demandas

capacidadMaxima, deposito, listaClientes, demandas = leer_archivo_vrp('./Dificil.vrp')
mapa_demandas = {cliente: demanda for cliente, demanda in zip(listaClientes, demandas)}

def obtener_demanda_ruta(ruta):
    return sum(mapa_demandas[cliente] for cliente in ruta)

def calculo_Total_Distancia(ruta):
    distancia_Total = 0
    punto_Actual = deposito
    for cliente in ruta:
        distancia_Total += np.linalg.norm(np.array(punto_Actual) - np.array(cliente))
        punto_Actual = cliente
    distancia_Total += np.linalg.norm(np.array(punto_Actual) - np.array(deposito))
    return distancia_Total

def graficar_ruta(ruta, generacion):
    plt.figure(figsize=(10, 6))
    for cluster in ruta:
        x = [c[0] for c in cluster]; y = [c[1] for c in cluster]
        color = np.random.rand(3,)
        plt.plot(x, y, marker='o', linestyle='--', color=color)
        if x: # Flechas deposito
            plt.plot([deposito[0], x[0]], [deposito[1], y[0]], color=color, linestyle='-')
            plt.plot([x[-1], deposito[0]], [y[-1], deposito[1]], color=color, linestyle='-')
    plt.plot(deposito[0], deposito[1], marker='D', color='red', markersize=10)
    plt.title(f"Generación {generacion} - Mejor Ruta"); plt.show()

# --- 2. HEURÍSTICA INICIAL (Igual que antes) ---
def calcular_Angulo(cliente):
    return math.atan2(cliente[1] - deposito[1], cliente[0] - deposito[0])

def heuristica_Barrido(listaClientes, demandas, capacidadMaxima):
    angulos= [(cliente, demanda, calcular_Angulo(cliente)) for cliente, demanda in zip(listaClientes, demandas)]
    angulos.sort(key = lambda x: x[2])
    clusters = []; cluster_Actual=[]; carga_Actual = 0
    for cliente, demanda, _ in angulos:
        if carga_Actual + demanda <= capacidadMaxima:
            cluster_Actual.append(cliente); carga_Actual += demanda
        else:
            clusters.append(cluster_Actual); cluster_Actual = [cliente]; carga_Actual = demanda
    if cluster_Actual: clusters.append(cluster_Actual)
    return clusters

clusters_iniciales = heuristica_Barrido(listaClientes, demandas, capacidadMaxima)

# --- 3. OPERADORES GENÉTICOS (Aquí están los cambios MÍNIMOS NECESARIOS) ---

def generador_de_PI(clusters, tamaño_Poblacion):
    return [[random.sample(c, len(c)) for c in clusters] for _ in range(tamaño_Poblacion)]

def seleccion_Torneo(poblacion, puntuaciones, k=3):
    seleccionados = random.sample(range(len(poblacion)), k)
    mejor = min(seleccionados, key=lambda i: puntuaciones[i])
    return poblacion[mejor]

# CAMBIO NECESARIO 1: Crossover que soporta rutas de diferente tamaño
def crossover_Robusto(padre, madre):
    """
    Crossover que inyecta una ruta de la madre en el padre,
    asegurando que NO se pierdan ni dupliquen clientes.
    """
    # 1. Clonamos al padre para usarlo de base
    hijo = copy.deepcopy(padre)
    
    # 2. Elegimos UNA ruta aleatoria de la madre para inyectar
    # (Esto ayuda a traer "buenas ideas" de agrupamiento de la madre)
    if not madre: return hijo
    ruta_madre = random.choice(madre)
    
    # Lista de clientes que vamos a insertar (los de esa ruta de la madre)
    clientes_a_insertar = set(tuple(c) for c in ruta_madre)
    
    # 3. Limpieza: Eliminamos esos clientes del 'hijo' (base padre)
    # Iteramos sobre todas las rutas del hijo
    for ruta in hijo:
        # Iteramos al revés para poder borrar elementos sin romper el índice del bucle
        for i in range(len(ruta) - 1, -1, -1):
            if tuple(ruta[i]) in clientes_a_insertar:
                del ruta[i]
    
    # 4. Inserción: Agregamos la ruta de la madre tal cual al hijo
    hijo.append(copy.deepcopy(ruta_madre))
    
    # 5. Limpieza final: Borramos rutas que hayan quedado vacías tras la extracción
    hijo = [ruta for ruta in hijo if len(ruta) > 0]
    
    return hijo

# CAMBIO NECESARIO 2: Mutación normal (orden)
def mutar_orden(individuo, prob=0.05):
    for ruta in individuo:
        if len(ruta) > 1 and random.random() < prob:
            i, j = random.sample(range(len(ruta)), 2)
            ruta[i], ruta[j] = ruta[j], ruta[i]

# CAMBIO NECESARIO 3: Mutación de TRANSFERENCIA (La clave para bajar de 2000)
def mutar_transferencia(individuo, prob=0.3):
    if random.random() > prob: return
    
    # Elegir ruta origen y destino
    rutas_idx = list(range(len(individuo)))
    if len(rutas_idx) < 2: return
    idx_ori, idx_des = random.sample(rutas_idx, 2)
    
    if not individuo[idx_ori]: return
    
    # Intentar mover cliente
    cliente = random.choice(individuo[idx_ori])
    if obtener_demanda_ruta(individuo[idx_des]) + mapa_demandas[cliente] <= capacidadMaxima:
        individuo[idx_ori].remove(cliente)
        individuo[idx_des].insert(random.randint(0, len(individuo[idx_des])), cliente)

# SA Simple con 2-Opt (Mejora local)
def SA_Simple(ruta, temp=100, iteraciones=10):
    if len(ruta) < 3: return ruta
    curr = ruta[:]
    best = ruta[:]
    cost_b = calculo_Total_Distancia(best)
    
    for _ in range(iteraciones):
        i, j = sorted(random.sample(range(len(curr)), 2))
        new_r = curr[:i] + curr[i:j+1][::-1] + curr[j+1:] # 2-Opt
        cost_n = calculo_Total_Distancia(new_r)
        
        if cost_n < cost_b or random.random() < math.exp(-(cost_n - calculo_Total_Distancia(curr)) / temp):
            curr = new_r
            if cost_n < cost_b: best = curr[:]; cost_b = cost_n
    return best

# --- 4. BUCLE PRINCIPAL SIMPLIFICADO ---
def algoritmo_Genetico_Simple(clusters, pop_size, max_no_improve):
    poblacion = generador_de_PI(clusters, pop_size)
    mejor_global_fit = float('inf')
    mejor_global_sol = None
    sin_mejora = 0
    
    generacion = 0
    while sin_mejora < max_no_improve:
        generacion += 1
        fits = [sum(calculo_Total_Distancia(r) for r in ind) for ind in poblacion]
        
        # Mejor local
        mejor_idx = np.argmin(fits)
        mejor_fit = fits[mejor_idx]
        
        print(f"Gen {generacion}: {mejor_fit:.2f} (Récord: {mejor_global_fit:.2f})")
        
        # Actualizar Global
        if mejor_fit < mejor_global_fit:
            mejor_global_fit = mejor_fit
            mejor_global_sol = copy.deepcopy(poblacion[mejor_idx])
            sin_mejora = 0
            print(f"--> ¡Nuevo Récord! {mejor_global_fit:.2f}")
        else:
            sin_mejora += 1
            
        # Selección y Cruce
        nueva_gen = []
        nueva_gen.append(copy.deepcopy(mejor_global_sol)) # Elitismo simple
        
        while len(nueva_gen) < pop_size:
            p1 = seleccion_Torneo(poblacion, fits)
            p2 = seleccion_Torneo(poblacion, fits)
            
            # Cruce Robusto
            h1 = crossover_Robusto(p1, p2)
            h2 = crossover_Robusto(p2, p1)
            
            # Mutaciones
            mutar_orden(h1)
            mutar_orden(h2)
            mutar_transferencia(h1) # ¡Vital!
            mutar_transferencia(h2) # ¡Vital!
            
            nueva_gen.extend([h1, h2])
            
        poblacion = nueva_gen[:pop_size]
        
        # SA esporádico (para refinar, sin lógicas complejas)
        if sin_mejora > 10:
             # Aplicar SA al mejor de la poblacion actual para intentar empujarlo
             idx = np.argmin(fits)
             candidato = poblacion[idx]
             for i in range(len(candidato)):
                 candidato[i] = SA_Simple(candidato[i])

    return mejor_global_sol, mejor_global_fit, generacion

# Ejecución
def leer_entero(mensaje, valor_por_defecto):
    """Lee un entero y usa un valor por defecto si se deja vacío."""
    entrada = input(f"{mensaje} [Default: {valor_por_defecto}]: ")
    if entrada.strip() == "":
        return valor_por_defecto
    try:
        return int(entrada)
    except ValueError:
        print(f"Entrada no válida. Usando valor por defecto: {valor_por_defecto}")
        return valor_por_defecto

# Pedir parámetros de forma segura
tam_pop = leer_entero("Tamaño Población (100-150)", 150)
max_no_imp = leer_entero("Max Generaciones sin mejora (50-100)", 80)

print(f"--> Iniciando con Población: {tam_pop}, Max Sin Mejora: {max_no_imp}")

mejor_ruta, mejor_score, gen = algoritmo_Genetico_Simple(clusters_iniciales, tam_pop, max_no_imp)
print(f"Final: {mejor_score:.2f}")
graficar_ruta(mejor_ruta, gen)
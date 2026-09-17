import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
    redundancia y exposición en ese orden.
    """
    coverage, redundancy, exposure = problem.score_components(configuration)

    return coverage - redundancy - exposure


def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
    usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
    mejoras aceptadas antes de retornar el OptimizationResult.
    """

    current = initial_configuration
    evaluations = 0
    iterations = 0
    
    current_score = configuration_score(problem, current)
    evaluations += 1
        
    history_configurations = [current]
    history_scores = [current_score]
    
    while iterations < max_iterations:
        best_neighbor = None
        best_score = current_score
        
        for neighbor in problem.neighbors(current):
            neighbor_score = configuration_score(problem, neighbor)
            evaluations += 1
            
            if neighbor_score > best_score:
                best_neighbor = neighbor
                best_score = neighbor_score
        
        if best_neighbor is None:
            break 
        
        current = best_neighbor
        current_score = best_score
        history_configurations.append(current)
        history_scores.append(current_score)
        
        iterations += 1
        

    return OptimizationResult(
        best_configuration=current,
        best_score=current_score,
        evaluations=evaluations,
        iterations=iterations,
        history=history_configurations,
        score_history=history_scores
    )


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.
    """
    return initial_temperature * (cooling_rate ** iteration)


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9
    current = initial_configuration
    current_score = configuration_score(problem, current)
    evaluations = 1

    best_configuration = current
    best_score = current_score

    history = [current]
    score_history = [current_score]

    temperature = initial_temperature
    iteration = 0

    while iteration < max_iterations:
        temperature = cooling_schedule(initial_temperature, cooling_rate, iteration)
        if temperature <= minimum_temperature:
            break

        candidate = rng.choice(problem.neighbors(current))
        candidate_score = configuration_score(problem, candidate)
        evaluations += 1

        delta = candidate_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_configuration = current
                best_score = current_score

        history.append(current)
        score_history.append(current_score)

        iteration += 1

    return OptimizationResult(
        best_configuration=best_configuration,
        best_score=best_score,
        evaluations=evaluations,
        iterations=iteration,
        history=history,
        score_history=score_history,
        metadata={
            "initial_temperature": initial_temperature,
            "cooling_rate": cooling_rate,
            "final_temperature": temperature,
        },
    )


def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """

    
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2
    #Usamos el pseudocodigo que sale en Notas de Clase IA (el libro)
    n= len(parent1)
    c= rng.randint(1, n-1) #rng usado para dar un numero aleatorio entero.
    child1 = parent1[:c] + parent2[c:]
    child2 = parent2[:c] + parent1[c:]

    return child1, child2


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    #Pseudocodigo de Notas de Clase IA pero solo aparece mutacion por inversion de bit y remplazo aleatorio de gen
    #aca es por intercambio entonces toca hacerle un retoque.
    if rng.random() < mutation_probability: #Caso donde toca mutar
        indices_unos= []
        indices_ceros= []
        for i, bit in enumerate(individual):
            if bit == 1:
                indices_unos.append(i)
            else:
                indices_ceros.append(i)

        if len(indices_unos) == 0 or len(indices_ceros) == 0:
            return individual #Caso 2 no hay intercambio.

        cromosoma= list(individual)
        i = rng.choice(indices_unos)
        j = rng.choice(indices_ceros)
        cromosoma[i]=0
        cromosoma[j]=1
        return tuple(cromosoma)
    else:
        return individual #Caso 3 no toca mutar.
   


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    poblacion = problem.initial_population(population_size, rng)
    scores = [configuration_score(problem, config) for config in poblacion]
    evaluations = len(poblacion)

    mejor_indice = scores.index(max(scores))
    mejor_global = poblacion[mejor_indice]
    mejor_global_score = scores[mejor_indice]

    history_configurations = [mejor_global]
    history_scores = [mejor_global_score]

    for gen in range(generations):
        ranked = sorted(zip(poblacion, scores), key=lambda item: item[1], reverse=True)
        nueva_poblacion = []
        for config, score in ranked[:elite_size]:
            nueva_poblacion.append(config) #Igual que con swap mutation.

        while len(nueva_poblacion) < population_size:
            padre1 = problem.tournament_select(poblacion, scores, rng)
            padre2 = problem.tournament_select(poblacion, scores, rng)
            hijo1, hijo2 = one_point_crossover(padre1, padre2, rng)
            hijo1 = problem.repair_configuration(hijo1, rng)
            hijo2 = problem.repair_configuration(hijo2, rng)
            hijo1 = swap_mutation(hijo1, mutation_probability, rng)
            hijo2 = swap_mutation(hijo2, mutation_probability, rng)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < population_size:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion
        scores = []
        for config in poblacion:
            scores.append(configuration_score(problem, config))
        evaluations += len(poblacion)

        mejor_indice = scores.index(max(scores))
        mejor_actual = poblacion[mejor_indice]
        mejor_actual_score = scores[mejor_indice]

        if mejor_actual_score > mejor_global_score:
            mejor_global = mejor_actual
            mejor_global_score = mejor_actual_score

        history_configurations.append(mejor_global)
        history_scores.append(mejor_global_score)

    return OptimizationResult(
        best_configuration=mejor_global,
        best_score=mejor_global_score,
        evaluations=evaluations,
        iterations=generations,
        history=history_configurations,
        score_history=history_scores,
    )
    

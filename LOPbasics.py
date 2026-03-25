def fitness_function(solution, lop_instance):
    total_fitness = 0
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            u = solution[i]
            v = solution[j]
            total_fitness += lop_instance[u][v]
    return total_fitness
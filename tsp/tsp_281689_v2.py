# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 42
STEADY_STATE = 1000


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            cur_path = self.evaluate_solution(path)
            plt.title(f"Current path: {cur_path:,}")
            print(cur_path)
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def tweak(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        #temp = new_solution[i1]
        #new_solution[i1] = new_solution[i2]
        #new_solution[i2] = temp
        new_solution[i1:i2] = new_solution[i1:i2][::-1]
        p = np.random.random()
    return new_solution

def main():

    problem = Tsp(NUM_CITIES)

    generation = []
    costs = []
    best_so_far = None
    best_sol_so_far = None

    mu = 50
    rate = 2
    lamb = mu * rate

    for i in range(mu):
        solution = np.array(range(NUM_CITIES))
        np.random.shuffle(solution)
        solution_cost = problem.evaluate_solution(solution)
        generation.append(solution)
        costs.append(solution_cost)
        if best_so_far is None or solution_cost < best_so_far:
            best_so_far = solution_cost
            best_sol_so_far = solution
    
    generation = np.array(generation)

    problem.plot(best_sol_so_far)

    history = [(0, best_so_far)]
    steady_state = 0
    step = 0

    ind = 0

    while steady_state < STEADY_STATE:

        new_generation = generation.copy().tolist()
        new_costs = costs.copy()
        steady_state += 1
        step += 1
        
        for i in generation: #two generated for lambda
            #THIS WILL BE mu,lambda+mu
            for j in range(rate):
                new_solution = tweak(i, pm=.9)
                new_generation.append(new_solution)
                solution_cost = problem.evaluate_solution(new_solution)
                new_costs.append(solution_cost)
        
        #cambiare nell'estrazione del migliore per mu volte:

        while len(new_generation) is not mu:
            worst_val = None
            worst_sol_ind = None    
            for i in range(len(new_generation)):
                solution_cost = problem.evaluate_solution(np.array(new_generation[i]))
                if worst_val is None or solution_cost > worst_val:
                    worst_val = solution_cost
                    worst_sol_ind = i
            new_generation.pop(worst_sol_ind)
        generation = new_generation.copy()
        generation = np.array(generation)

        best = None
        best_sol = None

        for i in generation:
            val = problem.evaluate_solution(i)
            if best is None or val < best:
                best = val
                best_sol = i.copy()
        
        if best < best_so_far:
            best_so_far = best
            best_sol_so_far = best_sol
            steady_state = 0

        if steady_state % 50 == 0:
            print(steady_state)

        history.append((step, best_so_far))
    problem.plot(best_sol_so_far)
    

    '''
    while steady_state < STEADY_STATE:
        step += 1
        steady_state += 1
        new_solution = tweak(solution, pm=.5)
        new_solution_cost = problem.evaluate_solution(new_solution)
        if new_solution_cost < solution_cost:
            solution = new_solution
            solution_cost = new_solution_cost
            history.append((step, solution_cost))
            steady_state = 0
    problem.plot(solution)
    '''

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()

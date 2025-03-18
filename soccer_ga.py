import numpy as np
import pandas as pd # type: ignore

formations = [
    "5-4-1",
    "4-4-2",
    "4-3-3",
    "5-2-2-1",
    "5-2-3",
    "5-3-2",
    "4-5-1",
    "4-3-1-2",
    "4-3-2-1",
    "4-4-1-1",
    "4-2-3-1",
    "4-2-4",
    "4-2-2-2",
    "4-2-1-3",
    "4-1-4-1",
    "4-1-3-2",
    "4-1-2-3",
    "4-1-2-1-2",
    "3-5-2",
    "3-5-1-1",
    "3-4-3",
    "3-4-2-1",
    "3-4-1-2",
    "3-1-4-2"
]

def check_csv_empty(self, params=None):
    pass

def init_csv(self):
    pass

def generate_random_pop() -> list[int]:
    size = len(formations)
    pop = []
    for formation in formations:
        new_form = formation.replace('-', '')
        defense = int(new_form[0])
        if len(new_form) == 3:
            midfield = int(new_form[1])
            attack = int(new_form[2])
            pop.append([defense, midfield, attack])
        elif len(new_form) == 4:
            print(new_form)
        elif len(new_form) == 5:
            print("5:", new_form)
    return pop

def genetic_algorithm() -> str:
    pop = generate_random_pop()
    while len(pop) > 0:
        pass
    return ""

def select() -> tuple[str, str]:
    c1, c2 = "", ""
    return c1, c2

def mutate(chromosome1 : str, chromsome2 : str) -> str:
    return ""

def crossover(chromosome1 : str, chromsome2 : str) -> str:
    return ""

def eval_fitness(chromosome1 : str) -> str:
    return ""

if __name__ == "__main__":
    csv_file = 'data.csv'
    generate_random_pop()
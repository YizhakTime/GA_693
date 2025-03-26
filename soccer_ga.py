import numpy as np
import pandas as pd # type: ignore
from copy import deepcopy

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
        #https://stackoverflow.com/questions/40954324/how-to-remove-hyphens-from-a-list-of-strings
        new_form = formation.replace('-', '')
        if len(new_form) == 3:
            defense = int(new_form[0])
            midfield = int(new_form[1])
            attack = int(new_form[2])
            pop.append([defense, midfield, attack])

        elif len(new_form) == 4:
            if int(new_form[0]) == 4:
                defense = int(new_form[0])
                if new_form == '4312' or new_form == '4321' or new_form == '4222' or new_form == '4213' \
                    or new_form == '4141' or new_form == '4132':
                    midfield = int(new_form[1])+int(new_form[2])
                    attack = int(new_form[3])
                elif new_form == '4231' or new_form == '4123':
                    midfield = 3
                    attack = 3
            elif int(new_form[0]) == 5:
                defense = int(new_form[0])
                midfield = int(new_form[1])+int(new_form[2])
                attack = int(new_form[3])
            else:
                if new_form == '3511' or new_form == '3421':
                    defense = int(new_form[0])
                    midfield = int(new_form[1])
                    attack = int(new_form[2])+int(new_form[3])
                elif new_form == '3412' or new_form == '3142':
                    defense = int(new_form[0])
                    midfield = int(new_form[1])+int(new_form[2])
                    attack = int(new_form[3])                
            pop.append([defense, midfield, attack])
        
        elif len(new_form) == 5:
            defense = int(new_form[0])
            midfield = int(new_form[1])+int(new_form[2])+int(new_form[3])
            attack = int(new_form[4])
            pop.append([defense, midfield, attack])
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

def find_unique(pop : list[list[int]]) -> list[list[int]]:
    tmp = deepcopy(pop)
    for i in range(len(tmp)):
        for j in range(i, len(tmp)):
            print(tmp[i], tmp[j])
            if tmp[i] == tmp[j]:
                pop.pop(j)
    return pop

if __name__ == "__main__":
    csv_file = 'data.csv'
    pop = generate_random_pop()
    print(len(pop))
    find_unique(pop)
    # print(pop)
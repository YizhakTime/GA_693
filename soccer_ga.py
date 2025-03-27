import numpy as np
import pandas as pd 

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

def genetic_algorithm(pop: list[str], iterations: int=10) -> str:
    while len(pop) > 0:
        pass
    return ""

# tournament selection
def select() -> tuple[str, str]:
    c1, c2 = "", ""
    return c1, c2

#mutation rate of 0.1
def mutate(chromosome1: str, chromsome2: str, mutation: float=0.1) -> str:
    return ""

#single point crossover
def crossover(chromosome1: str, chromsome2: str) -> str:
    return ""

def replace_inds(pop: list[str], child1: str, child2: str) -> list[str]:
    return [""]

def eval_fitness(csv_file: str, chromosome1: str) -> str:
    df = pd.read_csv(csv_file)
    # https://www.geeksforgeeks.org/get-a-specific-row-in-a-given-pandas-dataframe/
    stats = df.loc[df['Formations'] ==  chromosome1]
    # return fitness (what data type, most likely float)
    return ""

def find_indices(pop: list[list[int]]) -> dict:
    d = dict()
    for i, ind in enumerate(pop):
        if tuple(ind) not in d:
            d[tuple(ind)] = [i]
        else:
            d[tuple(ind)].append(i)
    return d

def remove_duplicates(pop: list[list[int]]) -> list[list[int]]:
    ind_keys = find_indices(pop)
    new_pop = list()
    for ind in ind_keys:
        print(ind, ind_keys[ind])
        for i, elem in enumerate(ind_keys[ind]):
            if i == 0:
                new_pop.append(pop[elem])
    return new_pop

def generate_pop():
    L = ['3-5-2', '3-4-3', '4-3-3', '4-5-1', '4-2-3-1', '4-4-2', '5-3-2', '5-4-1']
    size = len(L)
    for i in range(size):
        L.append(L[i].replace('-', ''))
    del L[:8]
    return L

if __name__ == "__main__":
    csv_file = 'data.csv'
    pop = generate_pop()


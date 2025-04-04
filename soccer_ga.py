import numpy as np
import pandas as pd  # type: ignore
import time
import matplotlib.pyplot as plt # type: ignore

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

def generate_pop():
    L = ['3-5-2', '3-4-3', '4-3-3', '4-5-1', '4-2-3-1', '4-4-2', '5-3-2', '5-4-1', '4-4-1-1', '4-1-2-1-2']
    size = len(L)
    for i in range(size):
        L.append(L[i].replace('-', ''))
    del L[:10]
    return L

def get_fitness(pop: list[str], csv: str, weights: list[float]) -> list[float]:
    fitnesses = []
    for p in pop:
        # print(p)
        fitness = eval_fitness(csv_file=csv, chromosome=p, weights=weights)
        fitnesses.append(fitness)
    return fitnesses

def eval_fitness(csv_file: str, chromosome: str, weights: float) -> float:
    df = pd.read_csv(csv_file)
    # https://www.geeksforgeeks.org/get-a-specific-row-in-a-given-pandas-dataframe/
    # stats = df.loc[df['Formations'] ==  chromosome1]
    # return fitness (what data type, most likely float)
    df['Formations'] = df['Formations'].astype('string')
    idx = df['Formations'] == chromosome
    avg_goals_scored = np.mean(df.loc[idx, ['Goals scored']])
    avg_goals_conceded = np.mean(df.loc[idx, ['Goals conceded']])
    avg_shots_on_target = np.mean(df.loc[idx, ['Shots on target']])
    avg_total = np.mean(df.loc[idx, ['Total shots']])
    avg_poss = np.mean(df.loc[idx, ['Possession']])
    avg_pass = np.mean(df.loc[idx, ['Passing accuracy']])
    avg_offense = np.mean(df.loc[idx, ['Offensive duels won']])
    avg_pen_scored = np.mean(df.loc[idx, ['Penalties scored']])
    avg_pen_missed = np.mean(df.loc[idx, ['Penalties missed']])
    avg_num_corners =  np.mean(df.loc[idx, ['Number of corners']])
    avg_num_counter = np.mean(df.loc[idx, ['Number of counter attacks']])
    avg_free_kick = np.mean(df.loc[idx, ['Number of free kicks']])
    #print(avg_goals_scored, avg_goals_conceded, avg_shots_on_target, avg_total, avg_poss, avg_pass, avg_offense, avg_pen_scored, avg_pen_missed, avg_num_corners, avg_num_counter, avg_free_kick)
    return avg_goals_scored*(0.2)+avg_goals_conceded*(-0.2)+\
    avg_shots_on_target*(0.2)+avg_total*(0.02)+avg_poss*(0.03)+avg_pass*(0.1)+avg_offense*(0.1)+\
    avg_pen_scored*(0.1)+avg_pen_missed*(-0.2)+avg_num_corners*(0.05)+avg_num_counter*(0.1)+avg_free_kick*(0.1)

# tournament selection
def select(pop: list[str], fitness: list[float]) -> tuple[str, str]:
    total_fitness = sum(fitness)
    probs_fits = []
    for i, fit in enumerate(fitness):
        fit_n = fit/total_fitness
        probs_fits.append(fit_n)
    parents = []
    for iter in range(0, 2):
        prev = probs_fits[0]
        calc = prev
        rand_choice = np.random.random()
        cum_probs = []
        for i in range(len(probs_fits)):
            if i != 0:
                calc = probs_fits[i]+prev
            cum_probs.append(calc)
            prev = calc
        else:
            for i in range(len(cum_probs)):
                if rand_choice < cum_probs[i]:
                    index = i
                    break
            parents.append(index)
    return pop[parents[0]], pop[parents[1]]

#mutation rate of 0.1
def mutate(chromosome1: str, chromsome2: str, mutation: float=0.1) -> str:
    return ""

#single point crossover
def crossover(chromosome1: str, chromsome2: str) -> str:
    return ""

def check_convergence(file: str, max_fitness, pop: list[str]) -> str:
    for p in pop:
        fitness = eval_fitness(file, chromosome1=p)
        if fitness >= max_fitness:
            return True
    return False

def genetic_algorithm(pop: list[str], csv: str, generations: int=10, weights: float=0.2) -> tuple[str, float]:
    start = time.time_ns()
    for gen in range(generations):
        fits = get_fitness(pop, csv, weights=weights)
        for i in range(len(fits)):
            print(fits[i], pop[i])
    end = time.time_ns()
    time_ns = end-start
    total = time_ns/(10**9)
    return "", total

if __name__ == "__main__":
    csv_file = 'data.csv'
    pop = generate_pop()
    genetic_algorithm(pop=pop, csv=csv_file, generations=1, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
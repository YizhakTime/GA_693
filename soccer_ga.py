import numpy as np
import pandas as pd  # type: ignore
import time
import matplotlib.pyplot as plt # type: ignore
import random

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

def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def generate_pop():
    L = ['3-5-2', '3-4-3', '4-3-3', '4-5-1', '4-2-3-1', '4-4-2', '5-3-2', '5-4-1', '4-4-1-1', '4-1-2-1-2']
    size = len(L)
    for i in range(size):
        L.append(L[i].replace('-', ''))
    del L[:10]
    return L

def get_fitness(pop: list[str], csv: str, weights: list[float]) -> tuple[list[float], list[np.ndarray], np.ndarray]:
    fitnesses, all_m = [], []
    for p in pop:
        # print(p)
        fitness, metrics, weights = eval_fitness(csv_file=csv, chromosome=p, weights=weights)
        all_m.append(metrics)
        fitnesses.append(fitness)
    return fitnesses, all_m, weights

def eval_fitness(csv_file: str, chromosome: str, weights: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_file)
    # https://www.geeksforgeeks.org/get-a-specific-row-in-a-given-pandas-dataframe/
    # stats = df.loc[df['Formations'] ==  chromosome1]
    # return fitness (what data type, most likely float)
    metrics = []
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
    metrics.append(avg_goals_scored)
    metrics.append(avg_goals_conceded)
    metrics.append(avg_shots_on_target)
    metrics.append(avg_total)
    metrics.append(avg_poss)
    metrics.append(avg_pass)
    metrics.append(avg_offense)
    metrics.append(avg_pen_scored)
    metrics.append(avg_pen_missed)
    metrics.append(avg_num_corners)
    metrics.append(avg_num_counter)
    metrics.append(avg_free_kick)
    metrics = np.array(metrics)
    # print("met", metrics)
    weights = np.array(weights)
    fit = metrics*weights
    # print("mul", fit)
    total_fitness = np.sum(fit)
    # print("total", total_fitness)
    return total_fitness, metrics, weights
  
# tournament selection
def select(pop: list[str], fitness: list[float]) -> tuple[str, str, int, int]:
    # total_fitness = np.sum(fitness)
    total_fitness = np.nansum(fitness)
    # print("total_fitness", total_fitness)
    normalized_fits = np.array(fitness)/total_fitness
    # print("normalized", normalized_fits)
    parent1, parent2 = 0, 0
    while parent1 == parent2:
        # print("parents", parent1, parent2)
        cum_probs = np.cumsum(normalized_fits)
        # print("cum_probs", cum_probs)
        prob = np.random.random()
        # print("prob", prob)
        for index, pro in enumerate(cum_probs):
            if prob < pro:
                parent1 = index
                break
        prob = np.random.random()
        for i, pro in enumerate(cum_probs):
            prob = np.random.random()
            if prob < pro:
                parent2 = i
                break
    # print(parent1, parent2)
    return pop[parent1], pop[parent2], parent1, parent2

#single point crossover
def crossover(pop: list[str], parent1: str, parent2: str, p_c: float) -> tuple[str, str]:
    choice = np.random.choice([0, 1], p=[1-p_c, p_c])
    # might need to handle duplicates
    # print("choice", choice)
    choice = 1
    # https://stackoverflow.com/questions/10631473/str-object-does-not-support-item-assignment
    if choice:
        pop = list(random_permutation(pop))
        # print("Shuffled", pop)
        if len(parent1) == 3 and len(parent2) == 3:
            pos = np.random.randint(0, 3)
            tmp1 = list(parent1)
            tmp2 = list(parent2)
            tmp = tmp1[pos]
            tmp1[pos] = tmp2[pos]
            tmp2[pos] = tmp
            if tmp1[pos] != tmp2[pos]:
                for p in pop:
                    if p[pos] == tmp1[pos]:
                        if len(p) == len(tmp1) and p != parent1:
                            parent1 = p
                            break
                for p in pop:
                    if p[pos] == tmp2[pos]:
                        if len(p)  == len(tmp2) and p != parent2:
                            parent2 = p
                            break
        elif len(parent1) == 3 and len(parent2) == 4:
            pos = np.random.randint(0, 4)
            tmp1, tmp2 = list(parent1), list(parent2)
            if pos < len(parent1):
                tmp = tmp1[pos]
                tmp1[pos] = tmp2[pos]
                tmp2[pos] = tmp
                if tmp1[pos] != tmp2[pos]:
                    for p in pop:
                        if p[pos] == tmp1[pos]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if p[pos] == tmp2[pos]:
                            if p != parent2:
                                parent2 = p
                                break
            elif pos > len(parent1):
                pos2 = np.random.randint(0, 3)
                tmp = tmp1[pos2]
                tmp1[pos2] = tmp2[pos]
                tmp2[pos] = tmp
                if tmp1[pos2] != tmp2[pos]:
                    for p in pop:
                        if p[pos2] == tmp1[pos2]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == tmp2[pos]:
                                if p != parent2:
                                    parent2 = p
                                    break
        elif len(parent1) == 4 and len(parent2) == 3:
            pos = np.random.randint(0, 4)
            l1, l2 = list(parent1), list(parent2)
            if len(parent2) > pos:
                tmp = l1[pos]
                l1[pos] = l2[pos]
                l2[pos] = tmp
                if l1[pos] != l2[pos]:
                    for p in pop:
                        if p[pos] == l1[pos]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if p[pos] == l2[pos]:
                            if p != parent2:
                                parent2 = p
                                break
            else:
                index = np.random.randint(0, 3)
                tmp = l1[pos]
                l1[pos] = l2[index]
                l2[index] = tmp
                if l1[pos] != l2[index]:
                    for p in pop:
                        if p[index] == l2[index]:
                            if p != parent2:
                                parent2 = p
                                break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l1[pos]:
                                if p != parent1:
                                    parent1 = p
                                    break
        elif len(parent1) == 3 and len(parent2) == 5:
            pos = np.random.randint(0, 5)
            l1, l2 = list(parent1), list(parent2)
            if pos < len(parent1):
                tmp = l1[pos]
                l1[pos] = l2[pos]
                l2[pos] = tmp
                if l1[pos] != l2[pos]:
                    for p in pop:
                        if p[pos] == l1[pos]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if p[pos] == l2[pos]:
                            if p != parent2:
                                parent2 = p
                                break
            else:
                index = np.random.randint(0, 3)
                tmp = l2[pos]
                l2[pos] = l1[index]
                l1[index] = tmp
                if l1[index] != l2[pos]:
                    for p in pop:
                        if p[index] == l1[index]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l2[pos]:
                                if p != parent2:
                                    parent2 = p
                                    break
        elif len(parent1) == 5 and len(parent2) == 3:
            pos = np.random.randint(0, 5)
            l1, l2 = list(parent1), list(parent2)
            if pos < len(parent2):
                tmp = l1[pos]
                l1[pos] = l2[pos]
                l2[pos] = tmp
                if l1[pos] != l2[pos]:
                    for p in pop:
                        if p[pos] == l1[pos]:
                            if p != parent1:
                                parent1 = p
                                break
                    for p in pop:
                        if p[pos] == l2[pos]:
                            if p != parent2:
                                parent2 = p
                                break
            else:
                index = np.random.randint(0, 3)
                tmp = l1[pos]
                l1[pos] = l2[index]
                l2[index] = tmp
                if l2[index] != l1[pos]:
                    for p in pop:
                        if p[index] == l2[index]:
                            if p != parent2:
                                parent2 = p
                                break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l1[pos]:
                                if p != parent1:
                                    parent1 = p
                                    break
        elif len(parent1) == 4 and len(parent2) == 5:
            pos = np.random.randint(0, 5)
            l1, l2 = list(parent1), list(parent2)
            if pos < len(parent1):
                tmp = l1[pos]
                l1[pos] = l2[pos]
                l2[pos] = tmp
                if l1[pos] != l2[pos]:
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l1[pos]:
                                if p != parent1:
                                    parent1 = p
                                    break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l2[pos]:
                                if p != parent2:
                                    parent2 = p
                                    break
            else:
                index = np.random.randint(0, 4)
                tmp = l2[pos]
                l2[pos] = l1[index]
                l1[index] = tmp
                if l1[index] != l2[pos]:
                    for p in pop:
                        if index < len(p):
                            if p[index] == l1[index]:
                                if p != parent1:
                                    parent1 = p
                                    break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l2[pos]:
                                if p != parent2:
                                    parent2 = p
                                    break
        elif len(parent1) == 5 and len(parent2) == 4:
            pos = np.random.randint(0, 5)
            l1, l2 = list(parent1), list(parent2)
            if pos < len(parent2):
                tmp = l1[pos]
                l1[pos] = l2[pos]
                l2[pos] = tmp
                if l1[pos] != l2[pos]:
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l1[pos]:
                                if p != parent1:
                                    parent1 = p
                                    break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l2[pos]:
                                if p != parent2:
                                    parent2 = p
                                    break
            else:
                index = np.random.randint(0, 4)
                tmp = l2[index]
                l2[index] = l1[pos]
                l1[pos] = tmp
                if l1[pos] != l2[index]:
                    for p in pop:
                        if index < len(p):
                            if p[index] == l2[index]:
                                if p != parent2:
                                    parent2 = p
                                    break
                    for p in pop:
                        if pos < len(p):
                            if p[pos] == l1[pos]:
                                if p != parent1:
                                    parent1 = p
                                    break

        elif len(parent1) == 4 and len(parent2) == 4:
            # check if random index doesn't index into str thats smaller than index
            pos = np.random.randint(0, 4)
            tmp1 = list(parent1)
            tmp2 = list(parent2)
            tmp = tmp1[pos]
            tmp1[pos] = tmp2[pos]
            tmp2[pos] = tmp
            if tmp1[pos] != tmp2[pos]:
                for p in pop:
                    if pos < len(p):
                        if p[pos] == tmp1[pos]:
                            if p != parent1:
                                parent1 = p
                                break
                for p in pop:
                    if pos < len(p):
                        if p[pos] == tmp2[pos]:
                            if p != parent2:
                                parent2 = p
                                break
    return parent1, parent2

#mutation rate of 0.1
def mutate(pop: list[str], parent1: str, parent2: str, p_m: float=0.1) -> tuple[str, str, int, int]:
    choice = np.random.choice([0, 1], p=[1-p_m, p_m])
    if choice:
        # if mutation occurs
        while True:
            rand_ind = np.random.randint(0, len(pop))
            # print(rand_ind)
            rand_parent = np.random.randint(0, 2)
            if rand_parent:
                # if rand parent is 1, then replace 2nd parent
                parent2 = pop[rand_ind]
            else:
                # else if rand parent is 0, replace 1st parent
                parent1 = pop[rand_ind]
            if parent1 != parent2:
                break
    ind1, ind2 = 0, 0
    for i in range(len(pop)):
        if pop[i] == parent1:
            ind1 = i
        if pop[i] == parent2:
            ind2 = i

    return parent1, parent2, ind1, ind2

def change_weights(chromsome1: int, chromsome2: int, metrics: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    # np argpartition (find k max elements)
    # between both chromsomes, find max 3 metrics from their fitness and increase/decrease weights
    # for that metric
    metric1 = metrics[chromsome1]
    metric2 = metrics[chromsome2]
    fit1, fit2 = (metric1*weights), (metric2*weights)
    total_fitness1, total_fitness2 = np.sum(fit1), np.sum(fit2)
    if total_fitness1 < total_fitness2:
        met_arr = metrics[chromsome1]
        max_s, max_p, max_n = 0, 0, 0
        max_is, max_ip, max_in = 0, 0, 0
        for i in range(len(met_arr)):
            if max_s < met_arr[i]:
                max_s = met_arr[i]
                max_is = i
            elif max_p < met_arr[i] and max_p != max_s:
                max_p = met_arr[i]
                max_ip = i
            elif max_n < met_arr[i] and max_n != max_p:
                max_n = met_arr[i]
                max_in = i
        # print(max_in, max_ip, max_is)
        increase = np.random.uniform(0, 0.1)
        weights[max_is]+=increase
        weights[max_ip]+=increase
        weights[max_in]+=increase
        decrease = np.random.uniform(0, 0.1)
        ml, mt, mz = met_arr[0], met_arr[0], met_arr[0]
        ml_ind, mt_ind, mz_ind = 0, 0, 0
        for i in range(len(met_arr)):
            if ml > met_arr[i]:
                ml = met_arr[i]
                ml_ind = 0
            elif mt > met_arr[i] and mt != ml:
                mt = met_arr[i]
                mt_ind = 0
            elif mz > met_arr[i] and mz != mt:
                mz = met_arr[i]
                mz_ind = 0

        for i in range(len(weights)):
            if i != max_in and i != max_ip and i != max_is:
                if i == ml_ind or i == mt_ind or i == mz_ind:
                    weights[i]-=decrease 
        #print("Weights", weights)
        return weights

    elif total_fitness1 > total_fitness2:
        met_arr = metrics[chromsome2]
        max_s, max_p, max_n = 0, 0, 0
        max_is, max_ip, max_in = 0, 0, 0
        for i in range(len(met_arr)):
            if max_s < met_arr[i]:
                max_s = met_arr[i]
                max_is = i
            elif max_p < met_arr[i] and max_p != max_s:
                max_p = met_arr[i]
                max_ip = i
            elif max_n < met_arr[i] and max_n != max_p:
                max_n = met_arr[i]
                max_in = i
        # print(max_in, max_ip, max_is)
        # weights = np.array(weights)
        increase = np.random.uniform(0, 0.1)
        weights[max_is]+=increase
        weights[max_ip]+=increase
        weights[max_in]+=increase
        decrease = np.random.uniform(0, 0.1)
        ###### new, decreased bottom 3 metrics
        ml, mt, mz = met_arr[0], met_arr[0], met_arr[0]
        ml_ind, mt_ind, mz_ind = 0, 0, 0
        for i in range(len(met_arr)):
            if ml > met_arr[i]:
                ml = met_arr[i]
                ml_ind = 0
            elif mt > met_arr[i] and mt != ml:
                mt = met_arr[i]
                mt_ind = 0
            elif mz > met_arr[i] and mz != mt:
                mz = met_arr[i]
                mz_ind = 0
        for i in range(len(weights)):
            if i != max_in and i != max_ip and i != max_is:
                if i == ml_ind or i == mt_ind or i == mz_ind:
                    weights[i]-=decrease 
        return weights

def find_top_formations(pop: list[str], fitnesses: np.ndarray) -> list[str]:
    max_f1, max_f2, max_f3 = 0, 0, 0
    max_f1_i, max_f2_i, max_f3_i = 0, 0, 0
    top = []
    for fit in range(len(fitnesses)):
        if max_f1 < fitnesses[fit]:
            max_f1 = fitnesses[fit]
            max_f1_i = fit
        elif max_f2 < fitnesses[fit] and max_f1 != max_f2:
            max_f2 = fitnesses[fit]
            max_f2_i = fit
        elif max_f3 < fitnesses[fit] and max_f2 != max_f3:
            max_f3 = fitnesses[fit]
            max_f3_i = fit
    top.append(pop[max_f1_i])
    top.append(pop[max_f2_i])
    top.append(pop[max_f3_i])
    return top, [max_f1, max_f2, max_f3]

def genetic_algorithm(pop: list[str], csv: str, \
        generations: int=10, p_c: float=0.6, p_m: float=0.1, weights: list[float] = [0.2, -0.2, 0.2, 0.02, 0.03, 0.1, 0.1, 0.1, -0.2, 0.05, 0.1, 0.1], x: np.ndarray = np.array([0, 1, 2])) -> tuple[list[str], float]:
    start = time.time_ns()
    gens, weighs, all_fits = [], [], []
    # print("weights", weights)
    for gen in range(generations):
        fits, metrics, weights = get_fitness(pop, csv, weights=weights)
        # print("fits", fits)
        # for i, m in enumerate(metrics):
        #     print(f"Metric {i}", m)
        # print("metrics", metrics)
        # print("weights2", weights)
        p1, p2, f1, f2 = select(pop=pop, fitness=fits)
        # print("selection", p1, p2, f1, f2)
        while p1 == p2:
            p1, p2 = crossover(pop=pop, parent1=p1, parent2=p2, p_c=p_c)
        # print("crossover", p1, p2)
        p1, p2, new_f1, new_f2 = mutate(pop=pop, parent1=p1, parent2=p2, p_m=p_m)
        # print("mutate", p1, p2, new_f1, new_f2)
        weights = change_weights(chromsome1=new_f1, chromsome2=new_f2, metrics=metrics, weights=weights)
        # print("update", weights)
        weighs.append(weights)
        gens.append(gen)
        all_fits.append(fits)
        # might need to calculate max fitness rn

    top_three, top_fits = find_top_formations(pop, fits)
    means_weights = np.mean(weighs, axis=1)
    mean_fits = np.mean(all_fits, axis=1)
    plt.plot(gens, mean_fits)
    plt.xlabel('Number of generations')
    plt.ylabel('Mean fitness')
    plt.title('Fitness over generations')
    plt.show()
    plt.plot(gens, means_weights)
    plt.xlabel('Number of generations')
    plt.ylabel('Mean weights')
    plt.title('Weights over generations')
    plt.show()
    colors = ['tab:red', 'tab:blue', 'tab:orange']
    fig, ax = plt.subplots(layout='constrained')
    rects = ax.bar(top_three, top_fits, label=top_three, color=colors)
    ax.bar_label(rects, padding=3)
    ax.set_ylabel('Fitness')
    ax.set_title('Top three formations')
    ax.set_xticks(x, top_three)
    ax.legend(title='Formations')
    plt.show()
    end = time.time_ns()
    time_ns = end-start
    total = time_ns/(10**9)
    return top_three, total

if __name__ == "__main__":
    csv_file = 'data.csv'
    weights = [0.2, -0.2, 0.2, 0.02, 0.03, 0.1, 0.1, 0.1, -0.2, 0.05, 0.1, 0.1]
    pop = generate_pop()
    top_dict = dict()
    generations = 5
    x_vals = np.array([0, 1, 2])
    for i in range(generations):
        top_3, total = genetic_algorithm(pop=pop, csv=csv_file, generations=500, p_c=0.5, p_m=0.01, weights=weights, x=x_vals)
        print("Total time", total)
        for i, top in enumerate(top_3):
            if top+'_'+str(i) not in top_dict:
                top_dict[top+'_'+str(i)] = 1
            else:
                top_dict[top+'_'+str(i)] += 1
    max_form = (max(top_dict, key=top_dict.__getitem__))
    # https://docs.python.org/3/howto/sorting.html#sortinghowto
    print("Max formation is", max_form.split('_')[0])
    # https://docs.python.org/3/library/stdtypes.html#string-methods

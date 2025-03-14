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

example = []

class Storage:
    def __init__(self, path):
        self._path = path

    def check_csv_empty(self, params=None):
        try:
            df = pd.read_csv('data.csv')
            return True
        except pd.errors.EmptyDataError:
            return False
    def init_csv(self):
        pass
        

def generate_random_pop() -> list[str]:
    return []

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

def main():
    csv_file = 'data.csv'
    db = Storage(csv_file)
    if db.check_csv_empty() is None:
        db.init_csv()

if __name__ == "__main__":
    main()

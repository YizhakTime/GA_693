import numpy as np
import pandas as pd # type: ignore

print("Hello")

time = 0

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
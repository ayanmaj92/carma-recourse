import torch
from abc import ABC, abstractmethod


class SEM(ABC):
    def __init__(self, functions, inverses, sem_name, label_eq=None):
        self.sem_name = sem_name

        if functions is None or inverses is None:
            raise NotImplementedError(f"SEM {sem_name}  not implemented.")

        self.functions = functions
        self.inverses = inverses
        self.label_eq = label_eq

    def adjacency(self):
        raise NotImplementedError

    def intervention_index_list(self):
        return [1]

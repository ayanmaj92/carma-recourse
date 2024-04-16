from .chain import Chain
from .chain_4 import Chain4
from .chain_5 import Chain5
from .triangle import Triangle
from .collider import Collider
from .fork import Fork
from .diamond import Diamond
from .simpson import Simpson
from .large_backdoor import LargeBackdoor
from .german_credit import GermanCredit
from .trianglesens import TriangleSens
from .collidersens import ColliderSens
from .chainsens import ChainSens
from .loan import Loan


sem_dict = {"chain": Chain, "chain-4": Chain4, "chain-5": Chain5, "triangle": Triangle, "collider": Collider,
            "fork": Fork, "diamond": Diamond, "simpson": Simpson, "large-backdoor": LargeBackdoor,
            "german": GermanCredit,
            "trianglesens": TriangleSens, "collidersens": ColliderSens, "chainsens": ChainSens,
            "loan": Loan
            }


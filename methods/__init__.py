from .fedavg import *
from .hyperfl import *


def local_update(rule):
    LocalUpdate = {'FedAvg': LocalUpdate_FedAvg,
                   'HyperFL': LocalUpdate_HyperFL,
                   }

    return LocalUpdate[rule]
from .fedavg import *
from .hyperfl import *
from .hyperfl_sparse import *   # new import

def local_update(rule):
    LocalUpdate = {'FedAvg': LocalUpdate_FedAvg,
                   'HyperFL': LocalUpdate_HyperFL,
                   'HyperFL_Sparse': LocalUpdate_HyperFL_Sparse,   # added
                   }

    return LocalUpdate[rule]


#  import numpy as np


class StringMatrix(object):
    def __init__(self, MJ, KJ, CJ, Ns):
        self.MJ = MJ  # np.zeros((Ns+1, Ns+1))
        self.KJ = KJ
        self.CJ = CJ
        self.Ns = Ns

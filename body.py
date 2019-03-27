import scipy.io
import numpy as np

class BodyMatrix(object):
    def __init__(self, MK, KK, CK, PhiB, Nb):
        self.MK = MK
        self.KK = KK
        self.CK = CK
        self.PhiB = PhiB
        self.Nb = Nb

    @classmethod
    def frommat(cls, filename):
        body_matrix = scipy.io.loadmat(filename, squeeze_me=True)
        body_matrix = body_matrix['body_matrix']
        MK = np.array(body_matrix['MK'].tolist())
        KK = np.array(body_matrix['KK'].tolist())
        CK = np.array(body_matrix['CK'].tolist())
        PhiB = np.array(body_matrix['PhiB'].tolist())
        Nb = int(body_matrix['Nb'])

        return cls(MK, KK, CK, PhiB, Nb)



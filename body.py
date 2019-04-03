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

    @classmethod
    def fromnp(cls, fmkfilename, phifilename):
        fmk = np.load(fmkfilename)
        phi = np.load(phifilename)

        freq = fmk[:, 0]
        mk = fmk[:, 1]
        xsi = np.ones(len(mk))*0.01  # fmk[:, 3]

        wdk = 2*np.pi*freq
        wnk = wdk/np.sqrt(1-xsi**2)
        ck = 2*mk*xsi*wnk
        kk = mk*wnk**2

        MK = np.diagflat(mk)
        KK = np.diagflat(kk)
        CK = np.diagflat(ck)

        PhiB = phi[:, 2]  # dz
        Nb = len(mk)

        return cls(MK, KK, CK, PhiB, Nb)

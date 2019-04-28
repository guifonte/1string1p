import scipy.io
import numpy as np


class BodyMatrix(object):
    def __init__(self, B1, B2, PhiBz, PhiBy, GBz, GBy, Nb):
        self.B1 = B1
        self.B2 = B2
        self.PhiBz = PhiBz
        self.PhiBy = PhiBy
        self.GBz = GBz
        self.GBy = GBy
        self.Nb = Nb

    @classmethod
    def frommat(cls, filename, dt):
        body_matrix = scipy.io.loadmat(filename, squeeze_me=True)
        body_matrix = body_matrix['body_matrix']
        MK = np.array(body_matrix['MK'].tolist())
        KK = np.array(body_matrix['KK'].tolist())
        CK = np.array(body_matrix['CK'].tolist())
        PhiBz = np.array(body_matrix['PhiBz'].tolist())
        PhiBy = np.array(body_matrix['PhiBy'].tolist())
        Nb = int(body_matrix['Nb'])

        BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
        B1 = BI @ (2 * MK / dt ** 2 - KK)
        B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
        B3 = BI

        GBz = B3 @ PhiBz
        GBy = B3 @ PhiBy

        return cls(B1, B2, PhiBz, PhiBy, GBz, GBy, Nb)

    @classmethod
    def fromnp(cls, fmkfilename, phifilename, dt):
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

        PhiBz = phi[:, 2]  # dz
        PhiBy = phi[:, 1]  # dy

        Nb = len(mk)

        BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
        B1 = BI @ (2 * MK / dt ** 2 - KK)
        B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
        B3 = BI

        GBz = B3 @ PhiBz
        GBy = B3 @ PhiBy

        return cls(B1, B2, PhiBz, PhiBy, GBz, GBy, Nb)

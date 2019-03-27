import numpy as np
import time


class SolverResult(object):
    def __init__(self, z_p, z_b1, z_b2, an, bn, Fc):
        self.z_p = z_p
        self.z_b1 = z_b1
        self.z_b2 = z_b2
        self.an = an
        self.bn = bn
        self.Fc = Fc


def solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs):
    start = time.time()
    # Simulation parameters
    dt = 1 / Fs
    t = np.linspace(0, np.floor(d / dt) - 1, np.floor(d / dt)) * dt

    # Input parameters

    # String
    MJ = string_matrix.MJ
    CJ = string_matrix.CJ
    KJ = string_matrix.KJ

    L = string_matrix.L

    Ns = string_matrix.Ns
    j = np.linspace(1, Ns, Ns)

    # Body
    MK = body_matrix.MK
    KK = body_matrix.KK
    CK = body_matrix.CK

    xb = L

    PhiB = body_matrix.PhiB
    Nb = body_matrix.Nb

    # Pluck parameters

    xp = pluck_parameters.xp
    Ti = pluck_parameters.Ti
    dp = pluck_parameters.dp
    F0 = pluck_parameters.F0
    Tr = Ti + dp

    # Initialisation

    # Modal quantities

    an = np.ones((Ns+1, len(t)), dtype=np.float64)
    bn = np.ones((Nb, len(t)), dtype=np.float64)
    an[:, (1, 2)] = 0
    bn[:, (1, 2)] = 0

    # Physical quantity
    Fc = np.zeros((1, len(t)), dtype=np.float64)
    Fc[:, (1, 2)] = 0

    # Finite difference scheme

    # String
    AI = np.linalg.inv(MJ / dt ** 2 + CJ / (2 * dt))
    A1 = AI @ (2 * MJ / dt ** 2 - KJ)
    A2 = AI @ (CJ / (2 * dt) - MJ / dt ** 2)
    A3 = AI

    # Body
    BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
    B1 = BI @ (2 * MK / dt ** 2 - KK)
    B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
    B3 = BI

    # Evaluation of string mode shapes

    PhiSc = np.zeros(len(j)+1)
    PhiSe = np.zeros(len(j)+1)

    PhiSc[0] = xb/L
    PhiSe[0] = xp/L

    for i in range(1, len(j)+1):
        PhiSc[i] = np.sin(j[i-1] * np.pi * xb / L)
        PhiSe[i] = np.sin(j[i-1] * np.pi * xp / L)

    # ITERATIVE COMPUTATION OF SOLUTION
    Fc = np.zeros(len(t))
    Fc1 = -(1 / (PhiSc @ A3 @ PhiSc.T + PhiB @ B3 @ PhiB.T))
    PhiBB1 = PhiB @ B1
    PhiBB2 = PhiB @ B2
    PhiScA1 = PhiSc @ A1
    PhiScA2 = PhiSc @ A2
    PhiScA3PhiSeT = PhiSc @ A3 @ PhiSe.T
    A3PhiSeT = A3 @ PhiSe.T
    A3PhiScT = A3 @ PhiSc.T
    B3PhiBT = B3 @ PhiB.T

    for i in range(1, len(t)-1):

        Fe = (F0 / dp) * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))

        Fc2 = (PhiBB1 @ bn[:, i] + PhiBB2 @ bn[:, i - 1] - PhiScA1 @ an[:, i] - PhiScA2 @ an[:, i-1] - PhiScA3PhiSeT * Fe)
        Fc[i] = Fc1 * Fc2

        '''print('i: {}   Fe: {}   Fc: {}   Fc1: {}   Fc2: {}'.format(i, Fe, Fc[i], Fc1, Fc2))'''

        # compute modal coordinates for next time - step

        an[:, i + 1] = A1 @ an[:, i] + A2 @ an[:, i - 1] + A3PhiSeT * Fe - A3PhiScT * Fc[i]

        bn[:, i + 1] = B1 @ bn[:, i] + B2 @ bn[:, i - 1] + B3PhiBT * Fc[i]

        if i % 1000 == 0:
            print("Progress: {} of {}".format(i, len(t)))
    #qu
    # String displacement
    end = time.time()
    print(end-start)
    z_p = np.real(PhiSe @ an)*1e3  # in mm
    z_b1 = np.real(PhiSc @ an)*1e3  # in mm

    # Body displacement
    z_b2 = np.real(PhiB @ bn)*1e3  # in mm

    return SolverResult(z_p, z_b1, z_b2, an, bn, Fc)

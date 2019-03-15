import numpy as np
import time
from SolverResult import SolverResult


def solverfd(body_matrix, string_matrix, string_parameters, pluck_parameters, d, Fs):
    start = time.time()
    # Simulation parameters
    dt = 1 / Fs
    t = np.linspace(0, np.floor(d / dt) - 1, np.floor(d / dt)) * dt

    # Input parameters

    # String
    MJ = np.array(string_matrix['MJ'].tolist())
    CJ = np.array(string_matrix['CJ'].tolist())
    KJ = np.array(string_matrix['KJ'].tolist())

    L = np.array(string_parameters['L'].tolist())

    Ns = int(string_matrix['Ns'])
    j = np.linspace(1, Ns, Ns)

    # Body
    MK = np.array(body_matrix['MK'].tolist())
    KK = np.array(body_matrix['KK'].tolist())
    CK = np.array(body_matrix['CK'].tolist())

    xb = L

    PhiB = np.array(body_matrix['PhiB'].tolist())
    Nb = len(PhiB)

    # Pluck parameters

    xp = np.array(pluck_parameters['xp'].tolist())
    Ti = np.array(pluck_parameters['Ti'].tolist())
    dp = np.array(pluck_parameters['dp'].tolist())
    F0 = np.array(pluck_parameters['F0'].tolist())
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
    Fc2 = np.float64(0)
    Fe = np.float64(0)

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

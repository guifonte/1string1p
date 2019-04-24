import numpy as np
import time
# from csolver import csolverfd1p


class SolverResult1p(object):
    def __init__(self, z_p, z_b1, z_b2, an, bn, Fc):
        self.z_p = z_p
        self.z_b1 = z_b1
        self.z_b2 = z_b2
        self.an = an
        self.bn = bn
        self.Fc = Fc


class SolverResult2p(object):
    def __init__(self, z_p, y_p, z_b1, z_b2, y_b1, y_b2, an, bn, Fc):
        self.z_p = z_p
        self.y_p = y_p
        self.z_b1 = z_b1
        self.z_b2 = z_b2
        self.y_b1 = y_b1
        self.y_b2 = y_b2
        self.an = an
        self.bn = bn
        self.Fc = Fc


'''def solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs, pol_num, cython_opt):
    if pol_num == 1:
        if cython_opt == 1:
            return csolverfd1p(body_matrix, string_matrix, pluck_parameters, d, Fs)
        else:
            return solverfd1p(body_matrix, string_matrix, pluck_parameters, d, Fs)
    else:
        return solverfd2p(body_matrix, string_matrix, pluck_parameters, d, Fs)'''


def solverfd1p(body_matrix, string_matrix, pluck_parameters, d, Fs):
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

    # an = np.ones((Ns+1, len(t)), dtype=np.float64)
    # bn = np.ones((Nb, len(t)), dtype=np.float64)

    # an[:, (0, 1)] = 0
    # bn[:, (0, 1)] = 0

    anf = np.ones((Ns + 1, int(len(t)/5)), dtype=np.float64)
    bnf = np.ones((Nb, int(len(t)/5)), dtype=np.float64)

    anf[:, 0] = 0
    bnf[:, 0] = 0
    ana = np.zeros(Ns + 1, dtype=np.float64)
    anb = np.zeros(Ns + 1, dtype=np.float64)
    bna = np.zeros(Nb, dtype=np.float64)
    bnb = np.zeros(Nb, dtype=np.float64)

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
    # Fc = np.zeros(len(t))

    Fc1 = -(1 / (PhiSc @ A3 @ PhiSc.T + PhiB @ B3 @ PhiB.T))
    PhiBB1 = PhiB @ B1
    PhiBB2 = PhiB @ B2
    PhiScA1 = PhiSc @ A1
    PhiScA2 = PhiSc @ A2
    PhiScA3PhiSeT = PhiSc @ A3 @ PhiSe.T
    A3PhiSeT = A3 @ PhiSe.T
    A3PhiScT = A3 @ PhiSc.T
    B3PhiBT = B3 @ PhiB.T
    F0_dp = F0/dp
    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)

    Fcf = np.zeros(int(len(t) / 5))

    for i in range(1, len(t)-1):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))

        # Fc2 = (PhiBB1@bn[:, i] + PhiBB2@bn[:, i - 1] - PhiScA1@an[:, i] - PhiScA2@an[:, i-1] - PhiScA3PhiSeT*Fe)

        # Fc[i] = Fc1 * Fc2

        # compute modal coordinates for next time - step

        # an[:, i + 1] = A1 @ an[:, i] + A2 @ an[:, i - 1] + A3PhiSeT * Fe - A3PhiScT * Fc[i]

        # bn[:, i + 1] = B1_diag * bn[:, i] + B2_diag * bn[:, i - 1] + B3PhiBT * Fc[i]

        if i % 2 == 0:
            Fc2temp = (PhiBB1 @ bna + PhiBB2 @ bnb - PhiScA1 @ ana - PhiScA2 @ anb - PhiScA3PhiSeT * Fe)
            Fctemp = Fc1 * Fc2temp
            anb = A1 @ ana + A2 @ anb + A3PhiSeT * Fe - A3PhiScT * Fctemp
            bnb = B1_diag * bna + B2_diag * bnb + B3PhiBT * Fctemp
        else:
            Fc2temp = (PhiBB1 @ bnb + PhiBB2 @ bna - PhiScA1 @ anb - PhiScA2 @ ana - PhiScA3PhiSeT * Fe)
            Fctemp = Fc1 * Fc2temp
            ana = A1 @ anb + A2 @ ana + A3PhiSeT * Fe - A3PhiScT * Fctemp
            bna = B1_diag * bnb + B2_diag * bna + B3PhiBT * Fctemp

        if i % 5 == 0:
            if i % 2 == 0:
                anf[:, int(i/5)] = anb
                bnf[:, int(i/5)] = bnb
            else:
                anf[:, int(i/5)] = ana
                bnf[:, int(i/5)] = bna

            Fcf[int(i/5)] = Fctemp

        if i % 20000 == 0:
            temp_end = time.time() - start
            print("Progress: {a:.2f}% - {b:.2f}s".format(a=((float(i) / len(t)) * 100), b=temp_end))

    # String displacement
    end = time.time()
    print("Elapsed time = {:.2f}s".format(end - start))
    z_p = np.real(PhiSe @ anf)*1e3  # in mm
    z_b1 = np.real(PhiSc @ anf)*1e3  # in mm

    # Body displacement
    z_b2 = np.real(PhiB @ bnf)*1e3  # in mm

    return SolverResult1p(z_p, z_b1, z_b2, anf, bnf, Fcf)


def solverfd2p(body_matrix, string_matrix, pluck_parameters, d, Fs):
    start = time.time()
    # Simulation parameters
    dt = 1 / Fs
    t = np.linspace(0, np.floor(d / dt) - 1, np.floor(d / dt)) * dt

    # Input parameters

    # String
    A = string_matrix.A
    GS = string_matrix.GS
    PHISc = string_matrix.PHISc

    L = string_matrix.L

    Ns = string_matrix.Ns
    j = np.linspace(1, Ns, Ns)

    # Body
    B = body_matrix.B
    GB = body_matrix.GB

    xb = L

    PHIB = body_matrix.PHIB
    Nb = body_matrix.Nb

    # Pluck parameters

    xp = pluck_parameters.xp
    Ti = pluck_parameters.Ti
    dp = pluck_parameters.dp
    F0 = pluck_parameters.F0
    gamma = pluck_parameters.gamma
    Tr = Ti + dp

    # Initialisation

    # Modal quantities

    a = np.ones((2*(Ns + 1), int(len(t)/5)), dtype=np.float64)
    b = np.ones((Nb, int(len(t)/5)), dtype=np.float64)

    a[:, 0] = 0
    b[:, 0] = 0
    a_1 = np.zeros(2*(Ns + 1), dtype=np.float64)
    a_2 = np.zeros(2*(Ns + 1), dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    Fc = np.zeros((2, int(len(t) / 5)), dtype=np.float64)
    Fc[:, (1, 2)] = 0

    # Evaluation of string mode shapes

    PhiSc = np.zeros(len(j)+1)
    PhiSe = np.zeros(len(j)+1)

    PhiSc[0] = xb/L
    PhiSe[0] = xp/L

    for i in range(1, len(j)+1):
        PhiSc[i] = np.sin(j[i-1] * np.pi * xb / L)
        PhiSe[i] = np.sin(j[i-1] * np.pi * xp / L)

    # ITERATIVE COMPUTATION OF SOLUTION

    F0_dp = F0/dp
    Fc1 = PHISc @ GS + PHIB @ GB
    PHIScA = PHISc @ A
    PHIBB = PHIB @ B
    for i in range(1, len(t)-1):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        FeZ = Fe*np.sin(gamma)
        FeY = Fe*np.cos(gamma)

        if i % 2 == 0:
            alpha = np.concatenate((a_1, a_2, FeZ, FeY), axis=1)
            beta = np.concatenate((b_1, b_2), axis=1)
            Fctemp = np.linalg.solve(Fc1, (PHIScA @ alpha - PHIBB @ beta))
            a_2 = A @ alpha - GS @ Fc[:, i]
            b_2 = B @ beta - GB @ Fc[:, i]
        else:
            alpha = np.concatenate((a_2, a_1, FeZ, FeY), axis=1)
            beta = np.concatenate((b_2, b_1), axis=1)
            Fctemp = np.linalg.solve(Fc1, (PHIScA @ alpha - PHIBB @ beta))
            a_1 = A @ alpha - GS @ Fc[:, i]
            b_1 = B @ beta - GB @ Fc[:, i]

        if i % 5 == 0:
            if i % 2 == 0:
                a[:, int(i/5)] = a_2
                b[:, int(i/5)] = b_2
            else:
                a[:, int(i/5)] = a_1
                b[:, int(i/5)] = b_1
            Fc[int(i/5)] = Fctemp

        if i % 20000 == 0:
            temp_end = time.time() - start
            print("Progress: {a:.2f}% - {b:.2f}s".format(a=((float(i) / len(t)) * 100), b=temp_end))

    end = time.time()
    print("Elapsed time = {:.2f}s".format(end - start))

    # String displacement at the plucking point
    z_p = np.real(PhiSe @ a[0:Ns, :])*1e3  # in mm
    y_p = np.real(PhiSe @ a[(Ns+1):(2*(Ns+1)-1), :]) * 1e3  # in mm

    # String displacement at the coupling point
    z_b1 = np.real(PhiSc @ a[0:Ns, :])*1e3  # in mm
    y_b1 = np.real(PhiSc @ a[(Ns+1):(2*(Ns+1)-1), :]) * 1e3  # in mm

    # Body displacement at the coupling point
    z_b2 = np.real(PHIB[1, :] @ b)*1e3  # in mm
    y_b2 = np.real(PHIB[2, :] @ b) * 1e3  # in mm

    return SolverResult2p(z_p, y_p, z_b1, z_b2, y_b1, y_b2, a, b, Fc)


def solverfd_1(body_matrix, string_matrix, pluck_parameters, d, Fs):
    start = time.time()
    # Simulation parameters
    dt = 1 / Fs
    t = np.linspace(0, np.floor(d / dt) - 1, np.floor(d / dt)) * dt

    # Input parameters

    # String
    A1 = string_matrix.A1
    A2 = string_matrix.A2
    GSe = string_matrix.GSe
    GSc = string_matrix.GSc
    PhiSe = string_matrix.PhiSe
    PhiSc = string_matrix.PhiSc
    L = string_matrix.L
    Ns = string_matrix.Ns

    j = np.linspace(1, Ns, Ns)

    # Body

    B1 = body_matrix.B1
    B2 = body_matrix.B2
    GBz = body_matrix.GBz
    PhiBz = body_matrix.PhiBz
    GBy = body_matrix.GBy
    PhiBy = body_matrix.PhiBy
    Nb = body_matrix.Nb
    xb = L

    # Pluck parameters

    xp = pluck_parameters.xp
    Ti = pluck_parameters.Ti
    dp = pluck_parameters.dp
    F0 = pluck_parameters.F0
    Tr = Ti + dp

    # Initialisation

    # Modal quantities

    anf = np.ones((Ns + 1, int(len(t)/5)), dtype=np.float64)
    bnf = np.ones((Nb, int(len(t)/5)), dtype=np.float64)

    anf[:, 0] = 0
    bnf[:, 0] = 0
    ana = np.zeros(Ns + 1, dtype=np.float64)
    anb = np.zeros(Ns + 1, dtype=np.float64)
    bna = np.zeros(Nb, dtype=np.float64)
    bnb = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    Fc = np.zeros((1, len(t)), dtype=np.float64)
    Fc[:, (1, 2)] = 0

    # ITERATIVE COMPUTATION OF SOLUTION

    Fc1 = -(1 / (PhiSc @ GSc + PhiBz @ PhiBz))
    PhiBB1 = PhiBz @ B1
    PhiBB2 = PhiBz @ B2
    PhiScA1 = PhiSc @ A1
    PhiScA2 = PhiSc @ A2
    PhiScGse = PhiSc @ GSe
    F0_dp = F0/dp
    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)

    Fcf = np.zeros(int(len(t) / 5))

    for i in range(1, len(t)-1):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))

        # Fc2 = (PhiBB1@bn[:, i] + PhiBB2@bn[:, i - 1] - PhiScA1@an[:, i] - PhiScA2@an[:, i-1] - PhiScA3PhiSeT*Fe)

        # Fc[i] = Fc1 * Fc2

        # compute modal coordinates for next time - step

        # an[:, i + 1] = A1 @ an[:, i] + A2 @ an[:, i - 1] + A3PhiSeT * Fe - A3PhiScT * Fc[i]

        # bn[:, i + 1] = B1_diag * bn[:, i] + B2_diag * bn[:, i - 1] + B3PhiBT * Fc[i]

        if i % 2 == 0:
            Fc2temp = (PhiBB1 @ bna + PhiBB2 @ bnb - PhiScA1 @ ana - PhiScA2 @ anb - PhiScGse * Fe)
            Fctemp = Fc1 * Fc2temp
            anb = A1 @ ana + A2 @ anb + GSe * Fe - GSc * Fctemp
            bnb = B1_diag * bna + B2_diag * bnb + GBz * Fctemp
        else:
            Fc2temp = (PhiBB1 @ bnb + PhiBB2 @ bna - PhiScA1 @ anb - PhiScA2 @ ana - PhiScGse * Fe)
            Fctemp = Fc1 * Fc2temp
            ana = A1 @ anb + A2 @ ana + GSe * Fe - GSc * Fctemp
            bna = B1_diag * bnb + B2_diag * bna + GBz * Fctemp

        if i % 5 == 0:
            if i % 2 == 0:
                anf[:, int(i/5)] = anb
                bnf[:, int(i/5)] = bnb
            else:
                anf[:, int(i/5)] = ana
                bnf[:, int(i/5)] = bna

            Fcf[int(i/5)] = Fctemp

        if i % 20000 == 0:
            temp_end = time.time() - start
            print("Progress: {a:.2f}% - {b:.2f}s".format(a=((float(i) / len(t)) * 100), b=temp_end))

    # String displacement
    end = time.time()
    print("Elapsed time = {:.2f}s".format(end - start))
    z_p = np.real(PhiSe @ anf)*1e3  # in mm
    z_b1 = np.real(PhiSc @ anf)*1e3  # in mm

    # Body displacement
    z_b2 = np.real(PhiBz @ bnf)*1e3  # in mm

    return SolverResult1p(z_p, z_b1, z_b2, anf, bnf, Fcf)



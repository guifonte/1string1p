import numpy as np
import time
# from csolver import csolverfd1p


class SolverResult(object):
    def __init__(self, z_p, z_b1, z_b2, y_p=None, y_b1=None, y_b2=None):
        self.z_p = z_p
        self.y_p = y_p
        self.z_b1 = z_b1
        self.z_b2 = z_b2
        self.y_b1 = y_b1
        self.y_b2 = y_b2


def solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs, pol_num, cython_opt):
    if pol_num == 1:
        return solverfd_1(body_matrix, string_matrix, pluck_parameters, d, Fs)
    else:
        return solverfd_2(body_matrix, string_matrix, pluck_parameters, d, Fs)


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
    Nb = body_matrix.Nb

    # Pluck parameters

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

    return SolverResult(z_p, z_b1, z_b2)


def solverfd_2(body_matrix, string_matrix, pluck_parameters, d, Fs):
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

    Ti = pluck_parameters.Ti
    dp = pluck_parameters.dp
    F0 = pluck_parameters.F0
    gamma = pluck_parameters.gamma
    Tr = Ti + dp

    # Initialisation

    # Modal quantities

    az = np.ones((Ns + 1, int(len(t)/5)), dtype=np.float64)
    ay = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    b = np.ones((Nb, int(len(t)/5)), dtype=np.float64)

    az[:, 0] = 0
    ay[:, 0] = 0
    b[:, 0] = 0

    az_1 = np.zeros(Ns + 1, dtype=np.float64)
    az_2 = np.zeros(Ns + 1, dtype=np.float64)
    ay_1 = np.zeros(Ns + 1, dtype=np.float64)
    ay_2 = np.zeros(Ns + 1, dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    Fcz = np.zeros(int(len(t) / 5))
    Fcy = np.zeros(int(len(t) / 5))

    # ITERATIVE COMPUTATION OF SOLUTION

    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)
    PhiScGSc = PhiSc @ GSc
    PhiB = np.vstack((PhiBz, PhiBy))
    GB = np.vstack((GBz, GBy))
    PhiBcGBc = PhiB @ GB.T
    PhiBcGBc[0, 0] = PhiBcGBc[0, 0] + PhiScGSc
    PhiBcGBc[1, 1] = PhiBcGBc[1, 1] + PhiScGSc
    Fc1 = np.linalg.inv(PhiBcGBc)

    F0_dp = F0/dp

    for i in range(1, len(t)-1):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        Fez = Fe * np.sin(gamma)
        Fey = Fe * np.cos(gamma)

        if i % 2 == 0:
            Aalphaz = A1 @ az_1 + A2 @ az_2 + GSe * Fez
            Aalphay = A1 @ ay_1 + A2 @ ay_2 + GSe * Fey
            Bbeta = B1_diag * b_1 + B2_diag * b_2
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fc2y = PhiSc @ Aalphay - PhiBy @ Bbeta
            Fcz_temp = Fc1[0, 0] * Fc2z + Fc1[0, 1] * Fc2y
            Fcy_temp = Fc1[1, 0] * Fc2z + Fc1[1, 1] * Fc2y

            az_2 = Aalphaz - GSc * Fcz_temp
            ay_2 = Aalphay - GSc * Fcy_temp
            b_2 = Bbeta + GBz * Fcz_temp + GBy * Fcy_temp

        else:
            Aalphaz = A1 @ az_2 + A2 @ az_1 + GSe * Fez
            Aalphay = A1 @ ay_2 + A2 @ ay_1 + GSe * Fey
            Bbeta = B1_diag * b_2 + B2_diag * b_1
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fc2y = PhiSc @ Aalphay - PhiBy @ Bbeta
            Fcz_temp = Fc1[0, 0] * Fc2z + Fc1[0, 1] * Fc2y
            Fcy_temp = Fc1[1, 0] * Fc2z + Fc1[1, 1] * Fc2y

            az_1 = Aalphaz - GSc * Fcz_temp
            ay_1 = Aalphay - GSc * Fcy_temp
            b_1 = Bbeta + GBz * Fcz_temp + GBy * Fcy_temp

        if i % 5 == 0:
            if i % 2 == 0:
                az[:, int(i / 5)] = az_2
                ay[:, int(i / 5)] = ay_2
                b[:, int(i / 5)] = b_2
            else:
                az[:, int(i / 5)] = az_1
                ay[:, int(i / 5)] = ay_1
                b[:, int(i / 5)] = b_1

            Fcz[int(i / 5)] = Fcz_temp
            Fcy[int(i / 5)] = Fcy_temp

        if i % 20000 == 0:
            temp_end = time.time() - start
            print("Progress: {a:.2f}% - {b:.2f}s".format(a=((float(i) / len(t)) * 100), b=temp_end))

        #  print("i: {a:d}   Fz: {b:.10f}   Fy: {c:.10f}".format(a=i, b=Fcz_temp, c=Fcy_temp))

    # String displacement
    end = time.time()
    print("Elapsed time = {:.2f}s".format(end - start))

    # String displacement at the plucking point
    z_p = np.real(PhiSe @ az) * 1e3  # in mm
    y_p = np.real(PhiSe @ ay) * 1e3  # in mm

    # String displacement at the coupling point
    z_b1 = np.real(PhiSc @ az) * 1e3  # in mm
    y_b1 = np.real(PhiSc @ ay) * 1e3  # in mm

    # Body displacement at the coupling point
    z_b2 = np.real(PhiBz @ b) * 1e3  # in mm
    y_b2 = np.real(PhiBy @ b) * 1e3  # in mm

    return SolverResult(z_p, z_b1, z_b2, y_p, y_b1, y_b2)


def derivative(a, dt):
    b = np.zeros(len(a) - 1)

    for i in range(1, len(a) - 1):
        b[i] = (a[i] - a[i-1])/dt

    return b

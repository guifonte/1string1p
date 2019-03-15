import numpy as np
from stringMatrix import StringMatrix


def strings(string_parameters,fhmax):

    L = string_parameters.L
    D = string_parameters.D
    mu = string_parameters.mu
    T = string_parameters.T
    B = string_parameters.B
    f0 = string_parameters.f0

    Ns = round(fhmax / f0 - 1)

    rho = 1.2  # air density - (Valette)
    neta = 1.8*10**(-5)  # air dynamic viscosity - (Valette)

    c = np.sqrt(T / mu)
    mj = np.ones(Ns)*L*mu/2  # modal masses
    kj = np.ones(Ns)  # modal stiffnesses
    cj = np.ones(Ns)  # modal damping
    Qd = 5500  # damping due to the dislocation phenomenom(Adjusted in Pathé) -
    # Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!

    # Qw =?? $ damping included for a wound string due to the dry friction between two consecutive turns.

    dVETE = 1 * 10 ** (-3)  # thermo and visco - elastic effects constant(Valette) -
    # It is a problem, high uncertainty in determination!

    for j in range(1, Ns):
        kj[j-1] = (j**2) * np.pi**2 * T / (2 * L) + (j**4) * B * np.pi**4 / (2 * L**3)

        omegaj = (j * np.pi / L) * np.sqrt(T / mu) * np.sqrt(1 + (j ** 2) * ((B * np.pi ** 2) / (T * L ** 2)))
        # modal angular frequencies(considering inharmonic effects)
        fj = omegaj / (2 * np.pi)

        # DAMPING PARAMETERS
        R = 2 * np.pi * neta + 2 * np.pi * D * np.sqrt(np.pi * neta * rho * fj)  # air friction coefficient
        Qf = (2 * np.pi * mu * fj) / R  # damping due to the air friction

        Qvt = ((T**2) * c)/((4 * np.pi ** 2) * B * dVETE * fj ** 2)  # damping due to the thermo and visco - elasticity

        Qt = (Qf * Qvt * Qd) / (Qvt * Qd + Qf * Qd + Qvt * Qf)  # Modal damping factor!

        cj[j-1] = (j * np.pi * np.sqrt(T * mu))/(2 * Qt)

    KJ = np.diagonal(np.array([T / L], kj))
    CJ = np.diagonal(np.array([0], cj))

    MJ = np.ones((Ns+1, Ns+1))
    MJ[0, 0] = L * mu / 3

    for j in range(1, Ns):
        MJ[0, j] = L * mu / (np.pi * j)
        MJ[j, 0] = MJ[0, j]

    MJ[1:-1, 1:-1] = np.diagonal(mj)

    string_matrix = StringMatrix(MJ, KJ, CJ, Ns)

    return string_matrix

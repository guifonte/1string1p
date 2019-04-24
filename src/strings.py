import numpy as np
import scipy.io
from scipy.linalg import block_diag


class StringParameters(object):
    def __init__(self, L, D, mu, T, B, f0, note):
        self.L = L
        self.D = D
        self.mu = mu
        self.T = T
        self.B = B
        self.f0 = f0
        self.note = note

    @classmethod
    def frommat(cls, filename):
        string_parameters = scipy.io.loadmat(filename, squeeze_me=True)
        string_parameters = string_parameters['string_parameters']
        L = float(string_parameters['L'])
        D = float(string_parameters['D'])
        mu = float(string_parameters['mu'])
        T = float(string_parameters['T'])
        B = float(string_parameters['B'])
        f0 = float(string_parameters['f0'])
        note = string_parameters['note']
        return cls(L, D, mu, T, B, f0, note)


class StringMatrix_1p(object):
    def __init__(self, MJ, KJ, CJ, L, Ns):
        self.MJ = MJ  # np.zeros((Ns+1, Ns+1))
        self.KJ = KJ
        self.CJ = CJ
        self.Ns = Ns
        self.L = L


class StringMatrix_2p(object):
    def __init__(self, A, GS, PHISc, L, Ns):
        self.A = A
        self.GS = GS
        self.PHISc = PHISc
        self.Ns = Ns
        self.L = L


class StringMatrix(object):
    def __init__(self, A1, A2, GSe, GSc, PhiSe, PhiSc, Ns, L):
        self.A1 = A1
        self.A2 = A2
        self.GSe = GSe
        self.GSc = GSc
        self.PhiSe = PhiSe
        self.PhiSc = PhiSc
        self.Ns = Ns
        self.L = L

def stringscalculator(string_parameters,fhmax, xp, dt, pol_num):

    if pol_num == 1:
        string_matrix = stringscalculator1p(string_parameters,fhmax)
    else:
        string_matrix = stringscalculator2p(string_parameters, fhmax, xp, dt)

    return string_matrix


def stringscalculator1p(string_parameters,fhmax):

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
    Qd = 5500  # damping due to the dislocation phenomenom(Adjusted in Pathe) -
    # Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!

    # Qw =?? $ damping included for a wound string due to the dry friction between two consecutive turns.

    dVETE = 1 * 10 ** (-3)  # thermo and visco - elastic effects constant(Valette) -
    # It is a problem, high uncertainty in determination!

    for j in range(1, Ns+1):
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

    kj = np.insert(kj, 0, T / L)
    KJ = np.diagflat(kj)

    cj = np.insert(cj, 0, 0)
    CJ = np.diagflat(cj)

    MJ = np.ones((Ns+1, Ns+1))
    MJ[0, 0] = L * mu / 3

    for j in range(1, Ns+1):
        MJ[0, j] = L * mu / (np.pi * j)
        MJ[j, 0] = MJ[0, j]

    mjdiag = np.diagflat(mj)
    MJ[1:, 1:] = mjdiag

    string_matrix = StringMatrix_1p(MJ, KJ, CJ, L, Ns)

    return string_matrix


def stringscalculator2p(string_parameters,fhmax, xp, dt):

    # String Properties
    L = string_parameters.L
    D = string_parameters.D
    mu = string_parameters.mu
    T = string_parameters.T
    B = string_parameters.B
    f0 = string_parameters.f0
    xp = xp
    xb = L
    Ns = round(fhmax / f0 - 1)

    rho = 1.2  # air density - (Valette)
    neta = 1.8*10**(-5)  # air dynamic viscosity - (Valette)

    c = np.sqrt(T / mu)
    mj = np.ones(Ns)*L*mu/2  # modal masses
    kj = np.ones(Ns)  # modal stiffnesses
    cj = np.ones(Ns)  # modal damping
    Qd = 5500  # damping due to the dislocation phenomenom(Adjusted in Pathe) -
    # Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!

    # Qw =?? $ damping included for a wound string due to the dry friction between two consecutive turns.

    dVETE = 1 * 10 ** (-3)  # thermo and visco - elastic effects constant(Valette) -
    # It is a problem, high uncertainty in determination!

    for j in range(1, Ns+1):
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

    kj = np.insert(kj, 0, T / L)
    KJ = np.diagflat(kj)

    cj = np.insert(cj, 0, 0)
    CJ = np.diagflat(cj)

    MJ = np.ones((Ns+1, Ns+1))
    MJ[0, 0] = L * mu / 3

    for j in range(1, Ns+1):
        MJ[0, j] = L * mu / (np.pi * j)
        MJ[j, 0] = MJ[0, j]

    mjdiag = np.diagflat(mj)
    MJ[1:, 1:] = mjdiag

    # Mode Shapes
    PhiSc = np.ones(Ns + 1)
    PhiSe = np.ones(Ns + 1)
    PhiSc[0] = xb/L;
    PhiSe[0] = xp/L;
    for j in range(1, Ns + 1):
        PhiSc[j] = np.sin(j * np.pi * xb / L)
        PhiSe[j] = np.sin(j * np.pi * xp / L)

    # Resolution Matrices
    AIZ = np.linalg.inv(MJ/dt**2 + CJ/(2*dt))
    A1Z = AIZ @ (2*MJ/dt**2 - KJ)
    A2Z = AIZ @ (CJ/(2*dt) - MJ/dt**2)
    A3Z = AIZ

    A1Y = A1Z
    A2Y = A2Z
    A3Y = A3Z

    A1 = block_diag(A1Z, A1Y)
    A2 = block_diag(A2Z, A2Y)
    A3 = block_diag(A3Z@PhiSe, A3Y@PhiSe)

    A = np.concatenate((A1, A2, A3.T), axis=1)
    GS = block_diag(A3Z @ PhiSc, A3Y @ PhiSc).T
    PHISc = block_diag(PhiSc , PhiSc)

    string_matrix = StringMatrix_2p(A, GS, PHISc, L, Ns)

    return string_matrix

def stringscalculator_1(string_parameters,fhmax, xp, dt):

    # String Properties
    L = string_parameters.L
    D = string_parameters.D
    mu = string_parameters.mu
    T = string_parameters.T
    B = string_parameters.B
    f0 = string_parameters.f0
    xp = xp
    xb = L
    Ns = round(fhmax / f0 - 1)

    rho = 1.2  # air density - (Valette)
    neta = 1.8*10**(-5)  # air dynamic viscosity - (Valette)

    c = np.sqrt(T / mu)
    mj = np.ones(Ns)*L*mu/2  # modal masses
    kj = np.ones(Ns)  # modal stiffnesses
    cj = np.ones(Ns)  # modal damping
    Qd = 5500  # damping due to the dislocation phenomenom(Adjusted in Pathe) -
    # Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!

    # Qw =?? $ damping included for a wound string due to the dry friction between two consecutive turns.

    dVETE = 1 * 10 ** (-3)  # thermo and visco - elastic effects constant(Valette) -
    # It is a problem, high uncertainty in determination!

    for j in range(1, Ns+1):
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

    kj = np.insert(kj, 0, T / L)
    KJ = np.diagflat(kj)

    cj = np.insert(cj, 0, 0)
    CJ = np.diagflat(cj)

    MJ = np.ones((Ns+1, Ns+1))
    MJ[0, 0] = L * mu / 3

    for j in range(1, Ns+1):
        MJ[0, j] = L * mu / (np.pi * j)
        MJ[j, 0] = MJ[0, j]

    mjdiag = np.diagflat(mj)
    MJ[1:, 1:] = mjdiag

    # Mode Shapes
    PhiSc = np.ones(Ns + 1)
    PhiSe = np.ones(Ns + 1)
    PhiSc[0] = xb/L;
    PhiSe[0] = xp/L;
    for j in range(1, Ns + 1):
        PhiSc[j] = np.sin(j * np.pi * xb / L)
        PhiSe[j] = np.sin(j * np.pi * xp / L)

    # Resolution Matrices
    AI = np.linalg.inv(MJ/dt**2 + CJ/(2*dt))
    A1 = AI @ (2*MJ/dt**2 - KJ)
    A2 = AI @ (CJ/(2*dt) - MJ/dt**2)
    A3 = AI

    GSe = (A3 @ PhiSe).T
    GSc = (A3 @ PhiSc).T

    string_matrix = StringMatrix(A1, A2, GSe, GSc, PhiSe, PhiSc, L, Ns)

    return string_matrix
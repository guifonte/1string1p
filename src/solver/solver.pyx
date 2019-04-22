import numpy as np
cimport numpy as np
cimport cython
import time

DTYPE = np.float
ctypedef np.float_t DTYPE_t

class SolverResult(object):
    def __init__(self, z_p, z_b1, z_b2, an, bn, Fc):
        self.z_p = z_p
        self.z_b1 = z_b1
        self.z_b2 = z_b2
        self.an = an
        self.bn = bn
        self.Fc = Fc

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs):
    start = time.time()
    cdef float temp_end
    # Simulation parameters
    dt = 1 / Fs
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] t = np.linspace(0, np.floor(d / dt) - 1, np.floor(d / dt)) * dt

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
    cdef float Ti = pluck_parameters.Ti
    dp = pluck_parameters.dp
    F0 = pluck_parameters.F0
    cdef float Tr = Ti + dp

    # Initialisation

    # Modal quantities

    cdef np.ndarray[DTYPE_t, ndim=2] an = np.ones((Ns + 1, int(len(t)/5)))
    cdef np.ndarray[DTYPE_t, ndim=2] bn = np.ones((Nb, int(len(t)/5)))

    an[:, 0] = 0
    bn[:, 0] = 0
    bn[:, 0] = 0

    # Auxiliary vectors for a[:,n], a[:,n-1], b[:,n] and b[:,n-1]
    cdef np.ndarray[DTYPE_t, ndim=1] ana = np.zeros(Ns + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] anb = np.zeros(Ns + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] bna = np.zeros(Nb)
    cdef np.ndarray[DTYPE_t, ndim=1] bnb = np.zeros(Nb)

    # Physical quantity
    cdef np.ndarray[DTYPE_t, ndim=1] Fc = np.zeros(int(len(t) / 5))

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

    cdef float Fc1 = -(1 / (PhiSc @ A3 @ PhiSc.T + PhiB @ B3 @ PhiB.T))
    cdef np.ndarray[DTYPE_t, ndim=1] PhiBB1 = PhiB @ B1
    cdef np.ndarray[DTYPE_t, ndim=1] PhiBB2 = PhiB @ B2
    cdef np.ndarray[DTYPE_t, ndim=1] PhiScA1 = PhiSc @ A1
    cdef np.ndarray[DTYPE_t, ndim=1] PhiScA2 = PhiSc @ A2
    cdef float PhiScA3PhiSeT = PhiSc @ A3 @ PhiSe.T
    cdef np.ndarray[DTYPE_t, ndim=1] A3PhiSeT = A3 @ PhiSe.T
    cdef np.ndarray[DTYPE_t, ndim=1] A3PhiScT = A3 @ PhiSc.T
    cdef np.ndarray[DTYPE_t, ndim=1] B3PhiBT = B3 @ PhiB.T
    cdef float F0_dp = F0/dp
    cdef np.ndarray[DTYPE_t, ndim=1] B1_diag = np.diagonal(B1)
    cdef np.ndarray[DTYPE_t, ndim=1] B2_diag = np.diagonal(B2)

    cdef float Fe
    cdef int endcount = len(t)-1
    cdef float Fc2temp
    cdef float Fctemp

    for i in range(1, endcount):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))

        # compute modal coordinates for next time - step

        # commute between the auxiliary vectors
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

        # Save only every 5 samples, due to the physical simulation Fs being 5 times the real Fs
        if i % 5 == 0:
            if i % 2 == 0:
                an[:, int(i/5)] = anb
                bn[:, int(i/5)] = bnb
            else:
                an[:, int(i/5)] = ana
                bn[:, int(i/5)] = bna

            Fc[int(i/5)] = Fctemp

        if i % 20000 == 0:
            temp_end = time.time() - start
            print("Progress: {a:.2f}% - {b:.2f}s".format(a=((float(i)/len(t))*100), b=temp_end))

    # String displacement
    end = time.time()
    print("Elapsed time = {:.2f}s".format(end-start))
    z_p = np.real(PhiSe @ an)*1e3  # in mm
    z_b1 = np.real(PhiSc @ an)*1e3  # in mm

    # Body displacement
    z_b2 = np.real(PhiB @ bn)*1e3  # in mm

    return SolverResult(z_p, z_b1, z_b2, an, bn, Fc)

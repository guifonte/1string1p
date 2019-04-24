import librosa

import time
from src.strings import stringscalculator, StringParameters, stringscalculator_1
from src.body import BodyMatrix1p, BodyMatrix2p, BodyMatrix
from src.pluck import PluckParameters
from src.solver.solver import solverfd_1

# Input parameters
string_path = './DATA/STRINGS/one_string_1.mat'
body_fmk_path = "./DATA/BODY/Ortho_plate_fmk"
body_phi_path = "./DATA/BODY/Ortho_plate_phi"
pluck_path = "./DATA/PLUCK/pluck_3.mat"

# Simulation parameters
fhmax = 5000  # Maximum frequency of string
Nb = 17
Fs = 5*44100
dt = 1/Fs
d = 3.2  # duration
pol_num = 2  # number of polarizations
cython_opt = 0  # uses cython

# Pluck parameters
# pluck_parameters = PluckParameters(0.50, 0.001, 0.008, 1, 0)
pluck_parameters = PluckParameters.frommat(pluck_path)


# String
string_parameters = StringParameters.frommat(string_path)
# string_matrix = stringscalculator(string_parameters, fhmax, pluck_parameters.xp, dt, pol_num)
string_matrix = stringscalculator_1(string_parameters, fhmax, pluck_parameters.xp, dt)

# Body
# body_matrix = BodyMatrix1p.frommat("./DATA/BODY/Viola_ComplexModes_Yzz_Yyz_NoNorm_matrix.mat")
'''if pol_num == 2:
    body_matrix = BodyMatrix1p.fromnp(body_fmk_path, body_phi_path)
else:
    body_matrix = BodyMatrix2p.frommat("./DATA/BODY/Viola_ComplexModes_Yzz_Yyz_NoNorm_matrix.mat")'''

body_matrix = BodyMatrix.frommat("./DATA/BODY/Viola_ComplexModes_Yzz_Yyz_NoNorm_17.mat", dt)

# Resolution
# result = solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs, pol_num, cython_opt)
result = solverfd_1(body_matrix, string_matrix, pluck_parameters, d, Fs)

timestamp = str(time.time()).split('.')[0]
librosa.output.write_wav('./OUT/out_'+timestamp+'.wav', result.z_p, sr=44100, norm=True)

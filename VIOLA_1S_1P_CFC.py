import librosa

import time
from src.strings import stringscalculator, StringParameters
from src.body import BodyMatrix
from src.pluck import PluckParameters
from src.solver.solver import solverfd

# Input parameters
stringpath = './DATA/STRINGS/one_string_1.mat'
bodyfmkpath = "./DATA/BODY/Ortho_plate_fmk"
bodyphipath = "./DATA/BODY/Ortho_plate_phi"
pluckpath = "./DATA/PLUCK/pluck_parameters_1.mat"

# Simulation parameters
fhmax = 20000  # Maximum frequency of string
Nb = 36
Fs = 5*44100
dt = 1/Fs
d = 4  # duration

# String
string_parameters = StringParameters.frommat(stringpath)
string_matrix = stringscalculator(string_parameters, fhmax)

# Body
# body_matrix = BodyMatrix.frommat("./DATA/BODY/Viola_ComplexModes_Yzz_Yyz_NoNorm_matrix.mat")
body_matrix = BodyMatrix.fromnp(bodyfmkpath, bodyphipath)

# Pluck parameters
# pluck_parameters = PluckParameters(0.50, 0.001, 0.008, 1, 0)
pluck_parameters = PluckParameters.frommat(pluckpath)

# Resolution
result = solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs)

timestamp = str(time.time()).split('.')[0]
librosa.output.write_wav('./OUT/out_'+timestamp+'.wav', result.z_p, sr=44100, norm=True)


from strings import stringscalculator, StringParameters
from body import BodyMatrix
from pluck import PluckParameters
from solver import solverfd
# Input parameters

# String
string_parameters = StringParameters.frommat('one_string_1.mat')

# Body
body_matrix = BodyMatrix.frommat("Viola_ComplexModes_Yzz_Yyz_NoNorm_matrix.mat")

# Pluck parameters
# pluck_parameters = PluckParameters(0.50, 0.001, 0.008, 1, 0)
pluck_parameters = PluckParameters.frommat("pluck_parameters_1.mat")

# Simulation parameters
fhmax = 20000  # Maximum frequency of string
Nb = 36
Fs = 5*44100
dt = 1/Fs
d = 8  # duration

string_matrix = stringscalculator(string_parameters, fhmax)

# Resolution

result = solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs)


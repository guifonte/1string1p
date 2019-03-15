import scipy.io
from Strings import strings
from solverFd import solverfd
from SolverResult import SolverResult

# Input parameters

# String
string_parameters = scipy.io.loadmat('one_string_1.mat', squeeze_me=True)
string_parameters = string_parameters['string_parameters']

# Body
body_matrix = scipy.io.loadmat("Viola_ComplexModes_Yzz_Yyz_NoNorm_matrix.mat", squeeze_me=True)
body_matrix = body_matrix['body_matrix']

# Pluck parameters
pluck_parameters = scipy.io.loadmat("pluck_parameters_1.mat", squeeze_me=True)
pluck_parameters = pluck_parameters['pluck_parameters']

# String Matrix
string_matrix = scipy.io.loadmat("one_string_1_matrix.mat", squeeze_me=True)
string_matrix = string_matrix['string_matrix']

# Simulation parameters
fhmax = 20000  # Maximum frequency of string
Nb = 36
Fs = 5*44100
dt = 1/Fs
d = 8  # duration

# Resolution

result = solverfd(body_matrix, string_matrix, string_parameters, pluck_parameters, d, Fs)


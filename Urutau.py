import librosa
import os
import time
import numpy as np
from src.strings import stringscalculator, StringParameters, stringscalculator_1
from src.body import BodyMatrix1p, BodyMatrix2p, BodyMatrix
from src.pluck import PluckParameters
from src.solver.solver import solverfd_2, derivative

# Input parameters
string_path = './DATA/STRINGS/one_string_1.mat'
body_path = "./DATA/BODY/orthoplate_complex_Modes_4_Node_23.mat"
body_fmk_path = "./DATA/BODY/Ortho_plate_fmk"
body_phi_path = "./DATA/BODY/Ortho_plate_phi"
pluck_path = "./DATA/PLUCK/pluck_3.mat"

# Simulation parameters
fhmax = 5000  # Maximum frequency of string
Fs = 5*44100
dt = 1/Fs
d = 8  # duration

pol_num = 2  # number of polarizations
cython_opt = 0  # uses cython
audio_pluck_disp = 0
audio_pluck_vel = 1
audio_pluck_acc = 0
audio_coupling_disp = 1
audio_coupling_vel = 1
audio_coupling_acc = 0


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
body_matrix = BodyMatrix.fromnp(body_fmk_path, body_phi_path, dt)
# body_matrix = BodyMatrix.frommat(body_path, dt)

# Resolution
# result = solverfd(body_matrix, string_matrix, pluck_parameters, d, Fs, pol_num, cython_opt)
result = solverfd_2(body_matrix, string_matrix, pluck_parameters, d, Fs)


timestamp = str(time.time()).split('.')[0]
infilename = os.path.splitext(os.path.basename(body_path))[0]
outpath = './OUT/' + timestamp
outfilename = infilename + '_' + string_parameters.note + '_' + str(fhmax) + 'Hz' + '.wav'

if audio_pluck_disp == 1:
    if pol_num == 1:
        pd = result.z_p
    else:
        pd = np.add(result.z_p, result.y_p)
    librosa.output.write_wav(outpath + '_pd_' + outfilename, pd, sr=44100, norm=True)

if audio_coupling_disp == 1:
    if pol_num == 1:
        cd = result.z_b1
    else:
        cd = np.add(result.z_b1, result.y_b1)
    librosa.output.write_wav(outpath + '_cd_' + outfilename, cd, sr=44100, norm=True)

if audio_pluck_vel == 1:
    if pol_num == 1:
        pv = derivative(result.z_p, dt)
    else:
        pv = np.add(derivative(result.z_p, dt), derivative(result.y_p, dt))
    librosa.output.write_wav(outpath + '_pv_' + outfilename, pv, sr=44100, norm=True)

if audio_coupling_vel == 1:
    if pol_num == 1:
        cv = derivative(result.z_b1, dt)
    else:
        cv = np.add(derivative(result.z_b1, dt), derivative(result.y_b1, dt))
    librosa.output.write_wav(outpath + '_cv_' + outfilename, cv, sr=44100, norm=True)

if audio_pluck_acc == 1:
    if pol_num == 1:
        pa = derivative(derivative(result.z_p, dt), dt)
    else:
        pa = np.add(derivative(derivative(result.z_p, dt), dt), derivative(derivative(result.y_p, dt), dt))
    librosa.output.write_wav(outpath + '_pa_' + outfilename, pa, sr=44100, norm=True)

if audio_coupling_acc == 1:
    if pol_num == 1:
        ca = derivative(derivative(result.z_b1, dt), dt)
    else:
        ca = np.add(derivative(derivative(result.z_b1, dt), dt), derivative(derivative(result.y_b1, dt), dt))
    librosa.output.write_wav(outpath + '_ca_' + outfilename, ca, sr=44100, norm=True)
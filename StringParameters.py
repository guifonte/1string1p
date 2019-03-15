import scipy.io


class StringParameters(object):
    def __init__(self, L, D, mu, T, B, f0):
        self.L = L
        self.D = D
        self.mu = mu
        self.T = T
        self.B = B
        self.f0 = f0

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

        return cls(L, D, mu, T, B, f0)

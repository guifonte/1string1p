import scipy.io


class PluckParameters1p(object):
    def __init__(self, xp, Ti, dp, F0, alpha):
        self.xp = xp
        self.Ti = Ti  # in (s)
        self.dp = dp
        self.F0 = F0
        self.alpha = alpha

    @classmethod
    def frommat(cls, filename):
        pluck_parameters = scipy.io.loadmat(filename, squeeze_me=True)
        pluck_parameters = pluck_parameters['pluck_parameters']
        xp = float(pluck_parameters['xp'])
        Ti = float(pluck_parameters['Ti'])
        dp = float(pluck_parameters['dp'])
        F0 = float(pluck_parameters['F0'])
        alpha = float(pluck_parameters['alpha'])

        return cls(xp, Ti, dp, F0, alpha)


class PluckParameters2p(object):
    def __init__(self, xp, Ti, dp, F0, gama):
        self.xp = xp
        self.Ti = Ti  # in (s)
        self.dp = dp
        self.F0 = F0
        self.gama = gama

    @classmethod
    def frommat(cls, filename):
        pluck_parameters = scipy.io.loadmat(filename, squeeze_me=True)
        pluck_parameters = pluck_parameters['pluck_parameters']
        xp = float(pluck_parameters['xp'])
        Ti = float(pluck_parameters['Ti'])
        dp = float(pluck_parameters['dp'])
        F0 = float(pluck_parameters['F0'])
        gama = float(pluck_parameters['gama'])

        return cls(xp, Ti, dp, F0, gama)

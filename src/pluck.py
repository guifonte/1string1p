import scipy.io


class PluckParameters(object):
    def __init__(self, xp, Ti, dp, F0, gamma):
        self.xp = xp  # position of the pluck in the string
        self.Ti = Ti  # starting point of the ramp in (s)
        self.dp = dp  # length of the ramp
        self.F0 = F0  # height of the ramp
        self.gamma = gamma  # incidence angle

    @classmethod
    def frommat(cls, filename):
        pluck_parameters = scipy.io.loadmat(filename, squeeze_me=True)
        pluck_parameters = pluck_parameters['pluck_parameters']
        xp = float(pluck_parameters['xp'])
        Ti = float(pluck_parameters['Ti'])
        dp = float(pluck_parameters['dp'])
        F0 = float(pluck_parameters['F0'])
        try:
            gamma = float(pluck_parameters['gamma'])
        except:
            gamma = None

        return cls(xp, Ti, dp, F0, gamma)

import numpy
import atmosphere

constants = {
    'g0': 9.80665, #NIST http://physics.nist.gov/Pubs/SP330/sp330.pdf
    'm': 5.972365e24, #wgs84
    'a': 6378137.00, #wgs84
    'omega': 7292115e-10, #wgs84
    'r0': 6356766.0, #USSA 1976
    'air': {
        'M': 28.9644, #USSA 1976
        'Cp': 1.005,
        'Cv': 0.718,
        'beta': 1.458e-6, #USSA 1976
        'S': 110.4, #USSA 1976
        'P0': 101325.0, #USSA 1976
        'rho0': 1.225, #USSA 1976
        'T0': 288.15, #USSA 1976
        'mu0': 1.7894e-5, #USSA 1976
        }
    }

class Atmosphere(atmosphere.GenericAtmosphere):
    """An atmosphere model for earth.
    """
    def __init__(self):
        """Constructor

        Arguments:
            no arguments

        Returns:
            class instance
        """
        super(Atmosphere, self).__init__()

        air = constants['air']
        self._M = air['M']
        self._Cp = air['Cp']
        self._Cv = air['Cv']

        self._beta = air['beta']
        self._S = air['S']

        self._T0 = air['T0']
        self._P0 = air['P0']

        self._g = constants['g0']
        self._r0 = constants['r0']

        self._compute_layer_props()

    def T_isa(self, z):
        """Compute the ISA temperature

        Valid between -1 and 47 km

        Arguments:
            z: altitude (m)

        Returns:
            T: temperature (K)
        """
        assert z > -1e3 and z < 47e3, 'z must be between -1 and 47 km'
        H = self.geopotential_height(z)
        layer = self._get_layer_props(H)

        return layer['T0'] + layer['lapse'] * (H - layer['H0'])

    def P_isa(self, z):
        """Compute the ISA pressure

        Valid between -1 and 47 km

        Arguments:
            z: altitude (m)

        Returns:
            P: pressure (Pa)
        """
        assert z > -1e3 and z < 47e3, 'z must be between -1 and 47 km'
        H = self.geopotential_height(z)
        layer = self._get_layer_props(H)

        if layer['lapse'] == 0.0:
            P = layer['P0'] * numpy.exp(
                -self._g / self.R / layer['T0'] * (H - layer['H0']))
        else:
            P = layer['P0'] * numpy.power(
                layer['T0'] / self.T_isa(z),
                self._g / self.R / layer['lapse'])

        return P

    def rho_isa(self, z):
        """Compute the ISA density

        Valid between -1 and 47 km

        Arguments:
            z: altitude (m)

        Returns:
            rho: density (kg/m3)
        """
        assert z > -1e3 and z < 47e3, 'z must be between -1 and 47 km'
        return self.P_isa(z) / self.R / self.T_isa(z)

    def mu_isa(self, z):
        """Compute the ISA dynamic viscosity

        Arguments:
            z: altitude (m)

        Returns:
            mu: dynamic viscosity (N-s/m2)
        """
        assert z > -1e3 and z < 47e3, 'z must be between -1 and 47 km'
        mu = self._beta * numpy.power(self.T_isa(z), 1.5) / (
            self.T_isa(z) + self._S)
        return mu

    def _compute_layer_props(self):
        """Compute the tables which we'll use for layer properties

        This should be called on initialization to prepare tables

        Arguments:
            no arguments

        Returns:
            no returns
        """
        H = numpy.array([0, 11000.0, 20000.0, 32000.0, 47000.0])
        lapse = [-6.5/1000.0, 0.0, 1.0/1000.0, 2.8/1000.0, 0.0]

        T = [self._T0,]
        P = [self._P0,]
        for idx, h in enumerate(H[1:]):
            T.append(T[-1] + (h - H[idx]) * lapse[idx])

            if lapse[idx] == 0.0:
                P_i = P[-1] * numpy.exp(
                    -self._g / self.R / T[-2] * (h - H[idx]))
            else:
                P_i = P[-1] * numpy.power(
                    T[-2] / T[-1],
                    self._g / self.R / lapse[idx])
            P.append(P_i)

        self._layers = {
            'H': H,
            'T': T,
            'P': P,
            'lapse': lapse}

    def _get_layer_props(self, H):
        """Get the properties defining each layer

        Arguments:
            H: geopotential height (m)

        Returns:
            props: dict of layer properties containing
                H0: layer base height
                T0: layer base temperature
                P0: layer base temperature
                lapse: layer lapse rate
        """
        idx = numpy.argmax(self._layers['H'] > H) - 1
        idx = numpy.clip(idx, 0, len(self._layers) - 1)
        props = {
            'H0': self._layers['H'][idx],
            'T0': self._layers['T'][idx],
            'P0': self._layers['P'][idx],
            'lapse': self._layers['lapse'][idx]}
        return props


import numpy

class GenericAtmosphere(object):
    """A Class for generic atmospheres
    """
    def __init__(self):
        """Constructor
        """

        self._R_bar = 8314.4598

        self._M = numpy.nan
        self._Cp = numpy.nan
        self._Cv = numpy.nan

        self._g = numpy.nan
        self._r0 = numpy.nan

    @property
    def R(self):
        """Compute the specific gas constant."""
        return self._R_bar / self._M

    @property
    def gamma(self):
        """Compute the ratio of specific heats."""
        return self._Cp / self._Cv

    @property
    def Cp(self):
        """Getter for specific heat at constant pressure."""
        return self._Cp

    @property
    def Cv(self):
        """Getter for specific heat at constant volume."""
        return self._Cv

    @property
    def M(self):
        """Getter for molar mass."""
        return self._M

    def g(self, z=0.0):
        """Compute the gravitational constant.

        From U.S. Standard Atmosphere, 1976

        Arguments:
            z: altitude

        Returns:
            g: gravitational constant adjusted for inverse-square law
        """
        g = self._g * numpy.power(self._r0 / (self._r0 + z), 2.0)
        return g

    def hypsometric(self, dz=None, T_bar=None, P1=None, P2=None, z=0.0):
        """Hypsometric Equation

        Any of the parameters in the hypsometric equation can be solved for.
        Leave the argument for the desired parameter blank (or pass in None)
        and it will be solved for as a function of the others.

        Arguments:
            dz: thickness of layer (m)
            T_bar: average temperature of the layer (K)
            P1: pressure at layer bottom (Pa)
            P2: pressure at layer top (Pa)
            z: base of the layer (for computing g, layer assumed thin enough
                that g does not vary significantly over it)

        Returns:
            whichever parameter was not specified
        """
        assert not (
                dz is not None and
                T_bar is not None and
                P1 is not None and
                P2 is not None), 'one parameter must be left free'

        if dz is None:
            assert T_bar is not None and P1 is not None and P2 is not None,\
                'only one parameter can be left free'
            return self.R * T_bar / self.g(z) * numpy.log(P1/P2)
        if T_bar is None:
            assert dz is not None and P1 is not None and P2 is not None,\
                'only one parameter can be left free'
            return dz / self.R * self.g(z) / numpy.log(P1/P2)
        if P1 is None:
            assert dz is not None and T_bar is not None and P2 is not None,\
                'only one parameter can be left free'
            return numpy.exp(dz / self.R * self.g(z)) * P2
        if P2 is None:
            assert dz is not None and T_bar is not None and P1 is not None,\
                'only one parameter can be left free'
            return numpy.exp(dz / self.R * self.g(z)) / P1

    def state(self, P=None, rho=None, T=None):
        """Equation of state

        This can solve for anything in the equation of state. You specify
        which to return by omitting an argument (or passing None).

        Arguments: leave one as none to solve for it
            P: pressure (Pa)
            rho: density (kg/m3)
            T: temperature (K)

        Returns:
            whichever of the inputs was not specified
        """
        assert not (P is not None and rho is not None and T is not None),\
            'one variable must be left free'
        if P is None:
            assert rho is not None and T is not None,\
                'only one variable can be left free'
            return rho * self.R * T
        if rho is None:
            assert P is not None and T is not None,\
                'only one variable can be left free'
            return P / self.R / T
        if T is None:
            assert P is not None and T is not None,\
                'only one variable can be left free'
            return P / self.R / rho

    def geopotential_height(self, z):
        """Compute the geopotential height of an altitude.

        Arguments:
            z: altitude (m)

        Returns:
            h: geopotential height (m)
        """
        return self._r0 * z / (self._r0 + z)



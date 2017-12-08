import pdb

import numpy
import copy

class DynamicSystem(object):
    def __init__(
        self,
        X0,
        dt,
        sys_dynamics_fcn,
        output_fcn=None,
        input_fcn=None):
        """ Constructor

        Arguments:
            X0: initial state
            dt: time step size
            sys_dynamics_fcn: function which defines the system dynamics. It
                takes in X and U and returns Xdot
            output_fcn: output function, takes X and returns Y
            input_fcn: a function to determine input from the state
                for example a controller

        Returns:
            object instance
        """
        self._X = X0
        self.Y = 0

        self._dynamics_fcn = sys_dynamics_fcn
        self._output_fcn = output_fcn
        self._input_fcn = input_fcn

        self._dt = dt

    def rk4(self, U=None):
        """ Integrate the system forward one timestep using an RK4 method

        Arguments:
            U: input, optional. If an input function was specified and this is
                not specified, then the input function will be used instead

        Returns:
            no returns
        """
        if U is None and self._input_fcn is None:
            U = self._input_fcn(self._X)
        assert U is not None, 'Either U or an input function must be specified'
        k1 = self._dynamics_fcn(self._X, U)
        k2 = self._dynamics_fcn(self._X + self._dt/2.0 * k1, U)
        k3 = self._dynamics_fcn(self._X + self._dt/2.0 * k2, U)
        k4 = self._dynamics_fcn(self._X + self._dt * k3, U)
        self._X += self._dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def euler(self, U=None):
        """ Integrate the system forward one timestep using an euler method

        Arguments:
            U: input, optional. If an input function was specified and this is
                not specified, then the input function will be used instead

        Returns:
            no returns
        """
        if U is None and self._input_fcn is None:
            U = self._input_fcn(self._X)
        assert U is not None, 'Either U or an input function must be specified'
        xdot = self._dynamics_fcn(self._X, U)
        self._X += self._dt * xdot

    def output(self):
        """ Compute the system output

        Arguments:
            no arguments

        Returns:
            Y: the system output
        """
        if self._output_fcn:
            return self._output_fcn(self._X)

    @property
    def state(self):
        """Getter for the state
        Arguments:
            no arguments

        Returns:
            state
        """
        return copy.deepcopy(self._X)

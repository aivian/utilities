# -*- coding: utf-8 -*-
import pdb
"""
Created on Fri May 22 11:11:49 2015

@author: Nate
Based heavily on the AVIA Raptor autopilot software origianlly written by Nate.
Also draws some elements from the software used on the Lionfish AUV written by
Nate and Albert Zheng.

This file holds PID control classes to be used in a variety of control
applications.
"""

# import numpy module
import numpy as np
from robot_control.states import ErrorState, ControlState
from robot_control.control import SISOControl

# pid-type controller base class
class PIDControl(SISOControl):
    """Class holding a general PID controller. This class is intended to be
    robust and to have a number of different use cases. Additionally, this
    class can represent P, PD, and PI controllers equally well."""
    def __init__(self, gains=[0.0]*3, state=0.0, signal=0.0,
                 output_method=None, output_filt=None, error_map=None):
        """Initialize the PIDControl class

        Args:
            gains: a three position list indicating the [p, i, d] gains for the
                controller. If only one argument is provided, the controller is
                assumed proportional only
            state: a ControlState object (but a float or int may also be
                provided, in which case we make a ControlState object here with
                the initial value provided) holding the state to be controlled
            signal: a ControlState object (same applies) holding the signal to
                be followed
            output_method: an optional method where the output of the
                controller is to be placed. It must take arguments:
                (cmd, time_now). Use this for directly setting a ControlState
                representing a signal to another controller.
            output_filt: optional output filter that can include rate limits
                (must contain a function called filter_value(value) that
                returns a float)
            error_map: optional mapping of the error before it is acted on
                (ex. deadband) must take an error and return a float
        """
        # initialize the parent class
        super(PIDControl, self).__init__(output_filt)
        # hold the state and signals
        if (type(state) is float) or (type(state) is int):
            # we were given a float, so we build a state object out of it
            self.state_obj = ControlState(state)
        else:
            self.state_obj = state
        # do the same for the signal
        if (type(signal) is float) or (type(signal) is int):
            # we were given a float, so the signal is constant (likely 0.0)
            self.signal_obj = ControlState(signal)
        else:
            self.signal_obj = signal
        # initialize an internal error state
        self.error_obj = ErrorState()
        # hold the error mapping if provided
        self._error_map = error_map
        # this method should take two arguments, a set_value and a time
        self.output_method = output_method
        # set gains [kp, ki, kd]
        self._gains = [0.0]*3
        if type(gains) is float:
            self.kp = gains
        elif len(gains) == 3:
            self._gains = gains
        # set my is_angle flag to match that of the state being controlled
        self._is_angle = self.state_obj._is_angle

    def _command_function(self, time):
        """Returns a command based on the current state and error values.

        Args:
            time: the time of the command call (seconds)

        Returns:
            a float representing the current control signal
        """
        # compute the current state error
        error = self.state_obj.get_error(self.signal_obj.value)
        # transform error
        if self._error_map is not None:
            error = self._error_map(error)
        # feed the error object
        #print "integral" , self.integral
        self.error_obj.set_value(error, time)

        #print "integral" , self.integral
        # if we've defined an output method
        if self.output_method is not None:
            # set the control signal in the appropriate method
            self.output_method(self.control_signal, time)

        # return the current control signal
        return self.control_signal

    @property
    def control_signal(self):
        """Returns the current control signal. This function should limit the
        output after computing it."""
        # compute the control value
        value = self.proportional + self.integral + self.state_derivative
        # limit and return
        # TODO: if clipping is required, clip integrated error as in C++ code
        # note that the integral cannot be clipped any further back than to zero
        return np.clip(value, self._limit[0], self._limit[1])

    @property
    def proportional(self):
        """Returns the proportional portion of a PID control output."""
        return self._gains[0]*self.error_obj.value

    @property
    def integral(self):
        """Returns the integrated error."""
        return self._gains[1]*self.error_obj.integral

    @property
    def error_derivative(self):
        """Returns the error derivative. Be careful using this as setpoint
        changes are a problem."""
        return self._gains[2]*self.error_obj.derivative

    @property
    def state_derivative(self):
        """Returns the negative of the state derivative."""
        return -1.0*self._gains[2]*self.state_obj.derivative

    def set_signal_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the command is allowed
        to take.

        Args:
            lower: float lower bound on the command
            upper: floar upper bound on the command
        """
        self.signal_obj.set_limit(lower, upper)

    def set_error_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the command is allowed
        to take.

        Args:
            lower: float lower bound on the command
            upper: floar upper bound on the command
        """
        self.error_obj.set_limit(lower, upper)

    def set_integrator_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the error integrator is
        allowed to reach.

        Args:
            lower: float lower bound on the integrator
            upper: floar upper bound on the integrator
        """
        self.error_obj.set_integral_limit(lower, upper)

    def set_state(self, state_input, time=0.0):
        """Method is used for callback to set states. The input argument
        can either be a float, in which case this method will have to be called
        frequently so that an updated state is always available, or it may
        be a handle to a ControlState object that represents the controlled
        state. In the later case, the state only needs to be supplied once (but
        the state itself must be updated!). The former use of this method is
        more or less depreciated and only still here for continuity. I think
        you should really be setting states in the state object itself.

        Arguments:
            state_input: state from the subscribed topic in the controller node
        """
        if isinstance(state_input, ControlState):
            # we've been handed a state object, use this going forward
            self.state_obj = state_input
        else:
            # we've been handed a value, send this to our state object
            self.state_obj.set_value(state_input, time)

    def set_state_derivative(self, state_deriv):
        """Method allows the setting of a state derivative. The input argument
        can either be a float, in which case this method will have to be called
        frequently so that an updated derivative is always available, or it may
        be a handle to a ControlState object that represents the controlled
        state's derivative. In the later case, ther derivative only needs to be
        supplied once (but the state itself must be updated!).

        Args:
            state_deriv: either a float or ControlState object
        """
        self.state_obj.derivative = state_deriv

    def set_ref_cmd(self, ref_cmd, time=0.0):
        """Method is used for callback to set the control target. The input
        argument can either be a float, in which case this method will have to
        be called frequently so that an updated target is always available, or
        it may be a handle to a ControlState object that represents the
        state target. In the later case, the target only needs to be supplied
        once (but the state itself must be updated!). The former use of this
        method is more or less depreciated and only still here for continuity.
        I think you should really be setting targets in the state object
        itself.

        Arguments:
            ref_cmd: ref_cmd from the subscribed topic in the controller node
        """
        if isinstance(ref_cmd, ControlState):
            # we've been handed a state object, use this going forward
            self.signal_obj = ref_cmd
        else:
            self.signal_obj.set_value(ref_cmd, time)

    def reset(self):
        """Method resets the controller. This is a safety method to prevent
        windup, etc."""
        # call the base controller reset
        super(PIDControl, self).reset()
        # reset the error object
        self.error_obj.reset()

    def ouput_gains(self):
        """Method returns the current set of gains for this control object."""
        return self._gains

    def set_gains(self, Kp, Ki, Kd):
        """Method is used to set gains either initialized in class or
        overwritten at a later time. This relies on the setter methods in the
        gain properties to do the error checking.
        """
        self.kp = Kp
        self.ki = Ki
        self.kd = Kd

    @property
    def kp(self):
        """Returns the proportional gain."""
        return self._gains[0]

    @kp.setter
    def kp(self, new_gain):
        """Sets the proportional gain but checks for unreasonable values."""
        if (np.abs(new_gain) < 1.0e6) and np.isfinite(new_gain):
            self._gains[0] = new_gain

    @property
    def ki(self):
        """Returns the integral gain."""
        return self._gains[1]

    @ki.setter
    def ki(self, new_gain):
        """Sets the integral gain but checks for unreasonable values."""
        if (np.abs(new_gain) < 1.0e6) and np.isfinite(new_gain):
            # clear any accumulated error if the old gain was zero
            if self._gains[1] == 0.0:
                # clear integrated error
                self.error_obj.integral = 0.0
            # rescale the integrated error to avoid jumps
            if np.abs(new_gain) > 0.0:
                self.error_obj.integral = (self.error_obj.integral *
                                           self._gains[1] / new_gain)
            # set the new gain
            self._gains[1] = new_gain

    @property
    def kd(self):
        """Returns the derivative gain."""
        return self._gains[2]

    @kd.setter
    def kd(self, new_gain):
        """Sets the derivative gain but checks for unreasonable values."""
        if (np.abs(new_gain) < 1.0e6) and np.isfinite(new_gain):
            self._gains[2] = new_gain

    @property
    def ti(self):
        """Returns the integral time?"""
        if np.abs(self.ki) > 0.0:
            return self.kp/self.ki
        else:
            return 0.0

    @ti.setter
    def ti(self, new_gain):
        """Sets the integral time."""
        if np.abs(new_gain) > 0.0:
            self.ki = self.kp/new_gain
        else:
            self.ki = 0.0

    @property
    def td(self):
        """Returns the derivative time."""
        if np.abs(self.kp) > 0.0:
            return self.kd/self.kp
        else:
            return 0.0

    @td.setter
    def td(self, new_gain):
        """Sets the derivative time."""
        self.kd = self.kp*new_gain

    @property
    def error(self):
        """Return the current error signal (setpoint - actual)."""
        return self.error_obj.value

    @error.setter
    def error(self, input_value):
        """Set the error that the controller acts on. I don't see a use for
        this outside of testing..."""
        self.error_obj.set_value(input_value)

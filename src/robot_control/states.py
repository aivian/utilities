# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:19:39 2015

@author: Nate
"""

# import numpy module
import numpy as np

def limit_angle(angle_list):
    """Function is used to sensibly limit an angle or a list of angles to the
    range [-pi, pi]

    Args:
        angle_list: either a single angle or a list of angles

    Returns:
        a single angle or a list of angles (of the same length as the input)
        but limited to the range [-pi, pi]
    """
    if type(angle_list) is list:
        output = []
        for angle in angle_list:
            if np.isfinite(angle):
                output.append(np.clip(angle, -1.0*np.pi, 1.0*np.pi))
            else:
                output.append(0.0)
        return output
    else:
        angle = angle_list
        if np.isfinite(angle):
            return np.clip(angle, -1.0*np.pi, 1.0*np.pi)
        else:
            return 0.0

def wrap_angle(angle_list):
    """Function is used to wrap an angle or a list of angles to the range
    [-pi, pi]

    Args:
        angle_list: either a single angle or a list of angles

    Returns:
        a single angle or a list of angles (of the same length as the input)
        but wrapped to the range [-pi, pi]
    """
    if type(angle_list) is list:
        output = []
        for angle in angle_list:
            if np.isfinite(angle):
                output.append((angle + np.pi)%(2.0 * np.pi) - np.pi)
            else:
                output.append(0.0)
        return output
    else:
        angle = angle_list
        if np.isfinite(angle):
            return (angle + np.pi)%(2.0 * np.pi) - np.pi
        else:
            return 0.0

class ControlBase(object):
    """Abstract base class for controllers and control states."""
    def __init__(self, initial_value=0.0, is_angle=False):
        # most recent time the state was updated (sec)
        self._time = 0.0
        # previous time the state was updated (sec)
        self._timek_1 = 0.0
        # current value of the object
        self._value = float(initial_value)
        # set the angle flag
        self._is_angle = is_angle
        # Set limits
        self._limit = [0.0]*2
        # set limits on the magnitude of the state
        self.set_limit(-1e6, 1e6)
        # hold a reset (or initial) value for the state
        self._initial_value = float(initial_value)

    def set_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the state is allowed to
        acheive. Typically this is a bit of nonsense except for states where
        for example a negative value is nonsense (eg. airspeed). This is also
        useful in limiting an error state to reasonable bounds.

        Args:
            lower: float lower bound of the state
            upper: floar upper bound of the state
        """
        self._limit = [lower, upper]
        if self._is_angle:
            self._limit = limit_angle(self._limit)

    def reset(self):
        """Method called to reset the control base, specifically the times."""
        # reset object value field
        self._value = np.clip(self._initial_value,
                              self._limit[0], self._limit[1])
        # reset times
        self._time = 0.0
        self._timek_1 = 0.0

    @property
    def dt(self):
        """Return the time difference between the most recent state
        settings."""
        return self._time - self._timek_1

    @property
    def value(self):
        """Return the current value of the object. This can and should be
        redefined in most inherited classes."""
        return self._value

    @value.setter
    def value(self, input_value):
        """Sets the value but without a timestamp. This method for setting the
        value is not recommended."""
        if np.isfinite(input_value):
            # hold the current input value
            self._value = np.clip(input_value, self._limit[0], self._limit[1])

class ControlState(ControlBase):
    """Class to hold and deal with the various properties associated with a
    system state used in a control framework. This state holds the current
    value, some history, functionality to determine the derivative (and
    optionally to filter the derivative), and the ability to compare it to a
    set value to determine an error."""
    # TODO: using derivative, project state forward in time get_value(time_now), get_dt(time_now)
    def __init__(self, initial_value=0.0, is_angle=False, derivative_filt=None,
                 derivative=None, bias=None):
        """Initialize the state class.

        Args:
            is_angle: boolean indicating whether the state is an angle and thus
                should be dealt with differently
            initial_value: float initial value of the state, defaults to 0.0
            derivative_filt: optional error derivative filter (must contain a
                function called filter_value(value) that returns a float)
            derivative: handle of another ControlState object that is this
                state's derivative (eg, phi = ControlState(True, p)
            bias: either a float or an handle to another ControlState
                representing an subtractive bias applied to this state
        """
        # initialize the base class
        super(ControlState, self).__init__(initial_value, is_angle)
        # current value of the state
        #self._value = float(initial_value)
        # previous value of the state
        self._valuek_1 = float(initial_value)
        # hold a bias for the value
        self._bias = 0.0
        # handle of a ControlState representing this state's subtractive bias
        self._bias_obj = bias
        # value of the current derivative
        self._derivative = 0.0
        # handle of a ControlState representing this state's derivative
        self._derivative_obj = derivative
        # optional derivative filter to be used when computing derivative internally
        self._derivative_filt = derivative_filt
        # flag indicating whether we're using a numerical derivative
        self._is_num_deriv = True
        # hold a reset (or initial) value for the state
        #self._initial_value = float(initial_value)

    def set_value(self, input_value, time=0.0):
        """Method sets the current value of the state and updates previous
        values. If a time is provided and a derivative is not otherwise
        provided, a numerical derivative is taken here as well."""
        if np.isfinite(input_value):
            # move the current time to the previous time spot
            self._timek_1 = self._time
            #save the new time
            self._time = time
            # use the setter method to set the current _value
            self.value = input_value
            # if we're using a numerical derivative, do this now
            if self._is_num_deriv and self.dt > 0.0:
                # TODO: if we wrapped the angle, special handling needed!
                if self._derivative_filt is None:
                    # we have no derivative filter, take raw
                    self._derivative = (self._value - self._valuek_1)/self.dt
                else:
                    # filter the derivative
                    pre_filt = (self._value - self._valuek_1)/self.dt
                    self._derivative = self._derivative_filt.filter_value(pre_filt)

    def get_error(self, comparison=0.0):
        """Returns the error between the current state value and a
        comparison."""
        if np.isfinite(comparison):
            # limit comparison
#            if self._is_angle:
#                comparison = limit_angle(comparison)
#            comparison = np.clip(comparison, self._limit[0], self._limit[1])
            # wrap and return error
            if self._is_angle:
                comparison = wrap_angle(comparison)
                return wrap_angle(comparison - self.value)
            else:
                return comparison - self.value
        else:
            return 0.0

    def reset(self):
        """Method resets the internal state of the state. Whoa man."""
        # reset base
        super(ControlState, self).reset()
        # reset state tracker
        #self._value = np.clip(self._initial_value,
        #                      self._limit[0], self._limit[1])
        self._valuek_1 = np.clip(self._initial_value,
                                 self._limit[0], self._limit[1])
        # reset derivative
        self._derivative = 0.0
        # and derivative filter if available
        if self._derivative_filt is not None:
            self._derivative_filt.reset()
        # reset bias
        self._bias = 0.0

    @property
    def value(self):
        """Return the current value of the state. Computed as the sum of the
        current value and an additive bias applied to the value."""
        return self._value - self.bias

    @value.setter
    def value(self, input_value):
        """Sets the value but without a timestamp. This method cannot compute a
        numerical derivative."""
        if np.isfinite(input_value):
            # move the current value to the previous value spot
            self._valuek_1 = self._value
            # check angle status
            if self._is_angle:
                input_value = wrap_angle(input_value)
            # hold the current input value
            self._value = np.clip(input_value, self._limit[0], self._limit[1])

    @property
    def initial_value(self):
        """Return the initial value of the state"""
        return self._initial_value

    @initial_value.setter
    def initial_value(self, input_value):
        """Sets the initial_value as well as the value and previous values"""
        if np.isfinite(input_value):
            # hold the current input value
            self._initial_value = float(input_value)
            # current value of the state
            self._value = float(input_value)
            # previous value of the state
            self._valuek_1 = float(input_value)

    @property
    def derivative(self):
        """Returns the current state derivative."""
        if self._derivative_obj is None:
            # if the derivative is stored internally, report it
            return self._derivative
        else:
            # if the derivative comes from another state object, get it
            return self._derivative_obj.value

    @derivative.setter
    def derivative(self, input_value):
        """Method sets the state derivative (as opposed to using the numerical
        derivative). Either a float value or an object may be provided"""
        if isinstance(input_value, ControlState):
            # we've been handed a handle to this state's derivative
            self._derivative_obj = input_value
        else:
            # set the numerical derivative flag false
            self._is_num_deriv = False
            # set the current derivative value
            if self._derivative_filt is None:
                # we have no derivative filter, take raw
                self._derivative = float(input_value)
            else:
                # filter the derivative
                self._derivative = self._derivative_filt.filter_value(input_value)

    @property
    def bias(self):
        """Returns the current value of the state bias."""
        if self._bias_obj is None:
            # we don't have a bias object, return the internally stored value
            return self._bias
        else:
            return self._bias_obj.value

    @bias.setter
    def bias(self, input_value):
        """Sets the current value of the state bias."""
        if isinstance(input_value, ControlState):
            # we've been handed a state object, use this going forward
            self._bias_obj = input_value
        else:
            self._bias = float(input_value)
            # we've been handed a value, send this to our state object
            #self._bias_obj.set_value(input_value)

class ErrorState(ControlState):
    """ErrorState class is used to keep track of error-specific things on top
    of the functionality already contained in the ControlState class."""
    def __init__(self, is_angle=False, derivative_filt=None):
        """Initialize a memeber of the ErrorState class.

        Args:
            is_angle: boolean indicating whether this is an angular error
            derivative_filt: optional error derivative filter (must contain a
                function called filter_value(value) that returns a float)
        """
        super(ErrorState, self).__init__(0.0, is_angle, derivative_filt)
        # hold onto the integrated error
        self._integral = 0.0
        # TODO: state rate limits
        # set limits on the magnitude of the integral
        self.set_integral_limit(-1e6, 1e6)

    def set_value(self, input_value, time=0.0):
        """Method sets the current value of the state and updates previous
        values. If a time is provided and a derivative is not otherwise
        provided, a numerical derivative is taken here as well."""
        # call the parent's implementation of this function
        super(ErrorState, self).set_value(input_value, time)
        # additionally, update the integral
        self._integral += (self._value + self._valuek_1)*self.dt/2.0
        # limit the size of the integral
        self._integral = np.clip(self._integral, self._integral_limit[0],
                                 self._integral_limit[1])

    def set_integral_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the state is allowed to
        acheive. Typically this is a bit of nonsense except for states where
        for example a negative value is nonsense (eg. airspeed). This is also
        useful in limiting an error state to reasonable bounds.

        Args:
            lower: float lower bound of the state
            upper: floar upper bound of the state
        """
        self._integral_limit = [lower, upper]
        # TODO: I don't know that this is valid here..
        if self._is_angle:
            self._integral_limit = limit_angle(self._integral_limit)

    def reset(self):
        """Method resets the internal state of the state. Whoa man."""
        # call the parent's implementation of this function
        super(ErrorState, self).reset()
        # and clear out the integrated error
        self._integral = 0.0

    @property
    def integral(self):
        """Return the integrated error."""
        return self._integral

    @integral.setter
    def integral(self, input_value):
        """Method is used to set the integral value. This is rarely used."""
        self._integral = np.clip(input_value, self._integral_limit[0],
                                 self._integral_limit[1])


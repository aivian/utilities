# -*- coding: utf-8 -*-
#        pass
"""
Created on Wed May 27 10:23:05 2015

@author: Nate

Code migrated from C++ library used on the AVIA Raptor autopilot with small
improvements, changes, and Pythonization.
"""

# import numpy module
import numpy as np
from robot_control.states import ControlBase

class SISOControl(ControlBase):
    """Abstract base class for a variety of single input / single output
    controllers. This holds common methods and properties required of all
    control objects."""
    def __init__(self, output_filt=None):
        """Initialize the siso control object.

        Args:
            output_filt: optional output filter that can include rate limits
                (must contain a function called filter_value(value) that
                returns a float)
        """
        # initialize the base class
        super(SISOControl, self).__init__()
        # add optional output filter
        self._output_filt = output_filt
        # TODO: from C++ code, add a mode or is_active flag

    def command(self, time=0.0):
        """Base command function called when this controller should return a
        command value.

        Args:
            time: current time in seconds

        Returns:
            a float command value that has been limited and optionally rate
                limtied and filtered.
        """
        # move the current time to the previous time spot
        self._timek_1 = self._time
        #save the new time
        self._time = time

        # TODO: a proportional controller is fine running without a time..
#        if self.dt <= 0.0:
#            return 0.0

        # run the function specific to this controller (this is already output
        # limited)
        command_value = self._command_function(time)

        # return the commanded value limited to output limits
        if self._output_filt is None:
            # return the raw command value
            self.value = command_value
        else:
            # return the output filtered value
            self.value = self._output_filt.filter_value(command_value)
        #return the current value of the control object
        return self.value

    def _command_function(self, time):
        """Abstract method, please redefine me! I hold the inherited
        controller's command function."""
        return 0.0

    def reset(self):
        """Method called to reset the controller."""
        # reset base
        super(SISOControl, self).reset()
        # reset the output filter if available
        if self._output_filt is not None:
            self._output_filt.reset()

    def set_output_limit(self, lower, upper):
        """Method sets the lower and upper bounds that the command is allowed
        to take. This method has the same effect as the inherited set_limit().

        Args:
            lower: float lower bound on the command
            upper: floar upper bound on the command
        """
        self.set_limit(lower, upper)

# functions used to optionally add some deadband to a controller's output

def deadband(half_band, center=0.0):
    """Function provides a deadband in a control or error.

    Args:
        half_band: half width of the desired deadband
        center: center of the deadband
    """
    return lambda x: (np.abs(x-center) - half_band)*np.sign(x-center)

def soft_deadband(half_band, center=0.0):
    """Function provides a deadband in a control or error.

    Args:
        half_band: approximate half width of the desired deadband
        center: center of the deadband
    """
    return lambda x: (np.abs(x-center)/half_band)**2.0*np.sign(x-center)

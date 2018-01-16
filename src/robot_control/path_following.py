""" A collection for path following controllers
"""
import pdb

import numpy

from geodesy.conversions import lla_to_ned
from geodesy.conversions import ned_to_lla
from geometry.lines import point_line_distance
from geometry.lines import get_point_on_line
import geometry.conversions
from geometry.helpers import ring_index

class PathFollowingController(object):
    """ Super-class
    """
    def __init__(self, path, is_ned=True, is_flat=True):
        """ Constructor

        Arguments:
            path: an nx3 numpy specifying positions along the path in 3-d space
            is_ned: optional flag indicating if the path is given in north,
                east, down coordinates. if false then lla is expected.

        Returns:
            object instance
        """

        assert type(path) is numpy.ndarray, "path must be numpy array"
        assert path.shape[1] == 3, "path must be nx3"

        super(PathFollowingController, self).__init__()
        self._is_ned = is_ned
        self._is_flat = is_flat
        self._path = path

        self._X = numpy.zeros((3,))

class ParkController(PathFollowingController):
    """ Class to implement the Park nonlinear path following controller
    """
    def __init__(self, path, L, is_ned=True, is_flat=True):
        """ Constructor

        Arguments:
            path: an nx3 numpy specifying positions along the path in 3-d space
            L: the lookahead distance on the path.
            is_ned: optional flag indicating if the path is given in north,
                east, down coordinates. if false then lla is expected.
            is_flat: optional flag indicating if the path is flat (ie, we don't
                expect vertical guidance from this controller)

        Returns:
            object instance
        """

        assert type(path) is numpy.ndarray, "path must be numpy array"
        assert path.shape[1] == 3, "path must be nx3"

        super(ParkController, self).__init__(path, is_ned, is_flat)
        self._L = L

        self._V = numpy.zeros((3,))

    def update_state(self, position, velocity):
        """ The controller requires knowledge of the aircraft state.

        This is intended to be run from a callback somewhere to the aircraft
        state message. Write your callback as required to pull out the
        relevant information and pass it to this function.

        Arguments:
            position: aircraft 3-d position, format must match the is_ned
                flag specified at construction
            velocity: aircraft 3-d velocity, must be in ned coordinates

        Returns:
            no returns
        """

        assert type(position) is numpy.ndarray, "position must be numpy array"
        assert type(velocity) is numpy.ndarray, "velocity must be numpy array"
        assert position.shape == (3,), "position must be 1,3 array"
        assert velocity.shape == (3,), "velocity must be 3, array"

        self._X = position
        if self._is_flat:
            self._X[2] = 0.0
        self._V = velocity

    def command(self, params=(True, True, False)):
        """ Call this when you want an acceleration command

        Options implemented as parameters for compatibility with the
        parameterized controller

        Arguments:
            params: tuple of parameters. Defaults (True, True)
                project_this_segment: optional boolean, if true then we'll
                    project the segment closest to the aircraft to generate
                    the path intersection point. If false, then we'll search
                    for a point that lies on the path itself.
                closed: boolean, whether the trajectory is closed or not.
                from_current_location: boolean, whether to go directly
                    from our current location to the next waypoint

        Returns:
            a_cmd: numpy 3, array of ned accelerations to steer the vehicle
                on to the desired trajectory
        """
        project_this_segment = params[0]
        closed = params[1]
        from_current_location = params[2]

        # convert to ned if required, either way we'll work with the trajectory
        # as relative to the current aircraft position
        if self._is_ned:
            traj = self._path - self._X
            v = self._V
        else:
            traj = lla_to_ned(self._path, numpy.array(self._X, ndmin=2))
            v = self._V
        x = numpy.zeros((3,))

        # if we have a flat trajectory then simply zero out the z components
        if self._is_flat:
            traj[:,2] = 0.0

        if closed:
            traj = numpy.vstack((traj, traj[0]))

        r,vertex,i = point_line_distance(x, traj, True)

        # we have two cases either we're further than our look-ahead from
        # the trajectory, or we're not... if we are then we want to ensure
        # that the acceleration is computed based on our desired look-ahead
        # not on the distance from the trajectory (or we might take forever
        # to turn toward it)
        if numpy.linalg.norm(r) > self._L:
            traj_vector = geometry.conversions.to_unit_vector(r) * self._L
        elif from_current_location:
            traj_vector = (
                geometry.conversions.to_unit_vector(vertex[1] - x) * self._L)
        else:
            # if we're projecting a segment then we simply need to know which
            # path segment we lie nearest.
            if project_this_segment:
                tangent = ring_index(traj, i+1) - ring_index(traj, i)
                tangent = tangent/numpy.linalg.norm(tangent)
                path_length = numpy.sqrt(self._L**2.0 - r.dot(r))
                traj_vector = x + r + tangent * path_length
            # in the second case we want to find the line segment where our
            # position projected on the trajectory falls in between the end
            # points. We'll then compute a vector from us to that line segment
            else:
                while numpy.linalg.norm(ring_index(traj, i+1)) < self._L:
                    i += 1
                traj_vector = get_point_on_line(
                    x, numpy.vstack(
                        (ring_index(traj, i), ring_index(traj, i+1))
                    ), self._L)[0]
                traj_vector = geometry.conversions.to_unit_vector(
                    traj_vector) * self._L

        # if we're navigating from our current location then the trajectory
        # vector is just L away on a line from us to the trajcetory
        if from_current_location:
            idx = (i + 1) % traj.shape[0]
            traj_vector = traj[idx] - x
            traj_vector = geometry.conversions.to_unit_vector(
                traj_vector) * self._L

        # then we simply compute the park law..
        a_cmd = 2/self._L**2.0 * numpy.cross(numpy.cross(v, traj_vector), v)
        return numpy.squeeze(a_cmd)

    @property
    def L(self):
        """Returns the look ahead distance."""
        return self._L

    @L.setter
    def L(self, set_value):
        """Protects the setting of the look ahead distance."""
        self._L = numpy.clip(set_value, 1.0e1, 1.0e3)

class FeedbackLinearizedParkController(ParkController):
    """ A class generalizing the park controller to accelerating trajectories

    The park controller is developed for linear trajectories (ie tracking a
    line from one point to another). Here we generalize that through the use
    of feedback linearization. A nominal acceleration is provided for every
    point along the trajectory and is interpolated between points. The returned
    acceleration command includes both the feedback term and this nominal term
    """

    def __init__(self, path, acceleration, L, is_ned=True, is_flat=True):
        """ Constructor

        Arguments:
            path: an nx3 numpy specifying positions along the path in 3-d space
            acceleration: an nx3 numpy array specifying the nominal
                acceleration at each point on the path
            L: the lookahead distance on the path.
            is_ned: optional flag indicating if the path is given in north,
                east, down coordinates. if false then lla is expected.
            is_flat: optional flag indicating if the path is flat (ie, we don't
                expect vertical guidance from this controller)

        Returns:
            object instance
        """

        super(FeedbackLinearizedParkController, self).__init__(
            path, L, is_ned, is_flat)

        assert type(acceleration) is numpy.ndarray, "acceleration must be numpy\
            array"
        assert acceleration.shape[1] == 3, "acceleration must be nx3"
        assert acceleration.shape == path.shape, "path and acceleration must\
            have the same size"

        self._accels = acceleration

    def command(self):
        """ Generate a command

        This will call the super-class method to generate the feedback and then
        do a linear interp along the current segment for the linearization

        Arguments:
            no arguments

        Returns:
            a_cmd: the sum of the linearization and feedback terms
        """

        a_feedback = ParkController.command(self)

        r,vertex,i = point_line_distance(self._X, self._path, True)

        near_point = self._X + r
        segment_position = (
            numpy.linalg.norm(near_point - vertex) /
            numpy.linalg.norm(ring_index(self._path, i+1) - vertex))

        a_linearization = (
            (1 - segment_position) * self._accels[i] +
            segment_position * self._accels[i+1])

        a_cmd = a_feedback + a_linearization

        return numpy.squeeze(a_cmd)

class ParameterizedParkController(ParkController):
    """ A park controller which uses a parameterized trajectory
    In general this will work for feedback linearization cases as well as
    simple trajectories. By specifying a linearization function which is
    uniformly zero then the pure feedback form is achieved.
    """
    def __init__(self, path, acceleration, L, is_ned=True, is_flat=True):
        """ Constructor

        Arguments:
            path: function handle, should take a tuple of parameters which
                will either be set at some point or passed when a command
                is required. A position and inertial velocity tupled will be
                handed to it as the first argument from the internal state of
                the controller, followed by the parameter tuple
            acceleration: function handle, should take the same tuple of
                parameters. Again, position and inertial velocity will be
                passed in as a tuple followed by the parameters.
            L: the lookahead distance on the path.
            is_ned: optional flag indicating if the path is given in north,
                east, down coordinates. if false then lla is expected.
            is_flat: optional flag indicating if the path is flat (ie, we don't
                expect vertical guidance from this controller)

        Returns:
            object instance
        """

        assert callable(path), "path must be a function"
        assert callable(acceleration), "acceleration must be a function"

        super(ParameterizedParkController, self).__init__(
            numpy.zeros((3,3)), L, is_ned, is_flat)

        self._path = path
        self._acceleration = acceleration
        self._params = None

    def command(self, params=None):
        """ Generate a steering command

        Arguments:
            params: optional parameters defining the trajectory to be followed
                if not specified then the internally saved parameters will be
                used. You must either set internal parameters or specify some!

        Returns:
            a_cmd: an acceleration command.
        """
        #if params is None:
        #    assert self._params is not None, "Parameters not set or passed in!"
        #    params = self._params

        a_linearization = self._acceleration((self._X, self._V), params)

        path, tangent = self._path((self._X, self._V), params)

        # we assume that if you are working in lla coordinates that the path
        # computations are all done relative to the aircraft location (ie,
        # the aircraft serves as the reference lla location)
        if self._is_ned:
            r = path - self._X
        else:
            r = path

        if numpy.linalg.norm(r) > self._L:
            # if we're too far from the path
            traj_vector = r/numpy.linalg.norm(r)*self._L
        else:
            # if we're near enough to the path
            # TODO: why was r.dot(r) previously negative here?
            # my intuition is this should be: numpy.sqrt(r.dot(r) + self._L**2.0 - numpy.dot(r,L))
            path_length = numpy.sqrt(r.dot(r.T) + self._L**2.0)
            traj_vector = r + tangent * path_length

        # park control law (note Vxtraj is same as V*sin(eta))
        a_feedback= (2.0/(self._L**2.0) *
                numpy.cross(numpy.cross(self._V, traj_vector), self._V))

        a_cmd = a_feedback + a_linearization

        return numpy.squeeze(a_cmd)

class CirclingParkController(ParameterizedParkController):
    """A parameterized path controller that is pre-built to do circles
    """
    def __init__(self, X0, R, L, direction=1, is_ned=True):
        """Constructor

        Arguments:
            X0: the circle center point numpy (1,3) array (lla or ned depending
                on the argument is_ned)
            R: the circle radius (m)
            L: the lookahead distance on the path. (m)
            direction: optional, turn direction, sign of the yaw rate. defaults
                to positive turn rate
            is_ned: optional flag indicating if the path is given in north,
                east, down coordinates. if false then lla is expected.

        Returns:
            class instance
        """
        assert isinstance(X0, numpy.ndarray), 'X0 must be numpy array'
        assert X0.shape == (1,3), 'X0 must be numpy (3,) array'

        super(CirclingParkController, self).__init__(
            self._circle,
            self._circle_accel,
            L,
            is_ned)
        self._X0 = X0
        self._R = R
        self._direction = numpy.sign(direction)

    def _circle(self, state, params=None):
        """Compute the path location and direction for the park controller

        A helper function, this will be called by an instance of
        ParameterizedParkController. It figures out where the nearest path
        location is relative to the aircraft and computes a vector that is
        along the path in the desired direction

        Arguments:
            state - (X,V):
                X: position
                V: inertial velocity of vehicle
            params: unused, here for compatibility

        Returns:
            (path, path_dot)
            path: closest point on the path
            path_dot: tangent to the path at that point
        """
        # compute the circle center relative to the aircraft
        if self._is_ned:
            # dx is a vector pointing to the circle center
            dx = self._X0[0] - state[0]
        else:
            dx = lla_to_ned(self._X0, numpy.array([state[0]]))[0]
        dx[2] = 0.0
        dx_hat = dx/numpy.linalg.norm(dx)

        # now figure out where the circle would lie and compute a vector
        # traveling around it in the proper direction
        path = dx - dx_hat * self._R + self._X0
        # unit vector tangent to circle pointing in desired direction of motion
        k_hat = numpy.array([0, 0, 1.0])
        path_dot = numpy.cross(dx_hat, self._direction * k_hat)
        return (path, path_dot)

    def _circle_accel(self, state, params=None):
        """ generate the acceleration required to stay parallel to this path

        A helper function, this will be called by an instance of
        ParameterizedParkController. It computes the acceleration required
        by the vehicle to remain parallel to the desired orbit (note that
        this is the acceleration required _At_its_distance_from_the_center_
        The error will be flown out by the park controller.

        Arguments:
            state - (X,V):
                X: numpy (3,) position
                V: numpy (3,) inertial velocity of vehicle
            params: unused, here for compatibility

        Returns:
            accel = numpy (3,) array indicating the required inertial
                acceleration
        """
        # compute the circle center relative to the aircraft
        if self._is_ned:
            dx = self._X0[0] - state[0]
        else:
            # TODO: unclear if the argument order is correct here
            dx = lla_to_ned(self._X0, numpy.array([state[0]]))[0]
        dx[2] = 0.0
        # find unit vector in direction of circle center
        dx_hat = dx / numpy.linalg.norm(dx)

        # find scalar acceleration (v^2/r)
        a_norm = state[1].dot(state[1]) / numpy.linalg.norm(dx)

        accel = dx_hat * a_norm
        return accel

    @property
    def R(self):
        """Getter for circle radius

        Arguments:
            no arguments

        Returns:
            R: the circle radius (m)
        """
        return self._R

    @R.setter
    def R(self, R):
        """Setter for the circle radius

        Arguments:
            R: the new circle radius

        Returns:
            no returns
        """
        self._R = R

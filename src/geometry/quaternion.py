"""
quaternion operations, written at some point and finished up on 25 Sep 2014

initialize using quaternion(x), defaults to a unit quaternion
representing a 0 rotation
John Bird
"""

from numpy import sin, cos, tan, arcsin, arccos, arctan2, cross
import numpy as np
import copy as copy


class Quaternion:
    """ quaternion class
    """
    def __init__(self, x=np.asarray([1.0, 0.0, 0.0, 0.0])):
        """ constructor

        Arguments:
            x: components, defaults to [1,0,0,0] (no rotation)

        Returns:
            the object
        """
        self.x = np.float64(np.asarray(x))

    def from_euler(self, euler):
        """ Set the quaternion from an euler array

        Arguments:
            euler: 3, numpy array of phi,theta,psi

        Returns:
            no returns
        """
        euler = np.float64(np.asarray(euler))
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]

        x = np.zeros((4))

        # equations for computing the quaternion components from euler angles
        x[0] = (cos(phi / 2.0) * cos(theta / 2.0) * cos(psi / 2.0) +
                sin(phi / 2.0) * sin(theta / 2.0) * sin(psi / 2.0))
        x[1] = (sin(phi / 2.0) * cos(theta / 2.0) * cos(psi / 2.0) -
                cos(phi / 2.0) * sin(theta / 2.0) * sin(psi / 2.0))
        x[2] = (cos(phi / 2.0) * sin(theta / 2.0) * cos(psi / 2.0) +
                sin(phi / 2.0) * cos(theta / 2.0) * sin(psi / 2.0))
        x[3] = (cos(phi / 2.0) * cos(theta / 2.0) * sin(psi / 2.0) -
                sin(phi / 2.0) * sin(theta / 2.0) * cos(psi / 2.0))

        self.x = x

    def __div__(self, other):
        """ Division operator for quaternions

        Arguments:
            other: the quaternion we're going to divide by this one

        Returns:
            other/this
        """
        assert isinstance(other, Quaternion)
        return other * self.inverse()

    def __mul__(self, other):
        """ multiplication operator for quaternions

        Arguments:
            other: the quaternion we're going to multiply by

        Returns:
            product
        """
        assert isinstance(other, Quaternion)

        x1 = self.x
        x2 = other.x
        q = copy.deepcopy(self)

        q.x[0] = x1[0] * x2[0] - np.dot(x1[1:4], x2[1:4])
        q.x[1:4] = x1[0] * x2[1:4] + x2[0] * \
            x1[1:4] + np.cross(x1[1:4], x2[1:4])
        return q

    def inverse(self):
        """ defines inverse for quaternion

        Arguments:
            no arguments

        Returns:
            no returns
        """
        q = copy.deepcopy(self)
        nq = q.norm()
        q.x[0] = q.x[0] / nq
        q.x[1:4] = -q.x[1:4] / nq
        return q

    def conjugate(self):
        """ defines conjugate of quaternion

        Arguments:
            no arguments

        Returns:
            q_star: the conjugate of this quaternion
        """
        q_star = Quaternion([self.x[0], -self.x[1], -self.x[2], -self.x[3]])
        return q_star

    def norm(self):
        """ Defines norm of quaternion

        Arguments:
            no arguments

        Returns:
            norm: the quaternion norm
        """
        return np.sqrt(np.sum(np.dot(self.x, self.x)))

    def normalize(self):
        """ Normalize the quaternion to a unit length

        The quaternion that this is called on will be modified such that it
        has a unit norm

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self.x /= self.norm()

    def rot(self, v, flag=True):
        """ Rotate a vector through the rotation represented by the quaternion.

        Arguments:
            v:  3, numpy array or something we can convert to it
            flag: defaults true, indicates which way we're going through the
                quaternion. If the orientation represents an aircraft then True
                indicates going from inertial to body axes

        Returns:
            the rotated vector
        """
        q = copy.deepcopy(self)
        q.x[0] = 0.0
        q.x[1:4] = np.float64(copy.deepcopy(v))

        if flag:
            qq = self.inverse() * q * self
        else:
            qq = self * q * self.inverse()

        return qq.x[1:4]

    def phi(self):
        """ Compute the equivalent phi angle for an euler triplet

        Arguments:
            no arguments

        Returns:
            phi: angle in radians
        """
        c23 = 2.0 * (self.x[2] * self.x[3] + self.x[0] * self.x[1])
        c33 = (pow(self.x[0], 2.0) - pow(self.x[1], 2.0) -
               pow(self.x[2], 2.0) + pow(self.x[3], 2.0))
        phi = arctan2(c23, c33)

        return phi

    def theta(self):
        """ Compute the equivalent theta angle for an euler triplet

        Arguments:
            no arguments

        Returns:
            theta: angle in radians
        """
        c13 = 2.0 * (self.x[1] * self.x[3] - self.x[0] * self.x[2])
        theta = -arcsin(c13)

        return theta

    def psi(self):
        """ Compute the equivalent psi angle for an euler triplet

        Arguments:
            no arguments

        Returns:
            psi: angle in radians
        """
        c12 = 2.0 * (self.x[1] * self.x[2] + self.x[0] * self.x[3])
        c11 = (pow(self.x[0], 2.0) + pow(self.x[1], 2.0) -
               pow(self.x[2], 2.0) - pow(self.x[3], 2.0))
        psi = arctan2(c12, c11)

        return psi

    def euler(self):
        """ Compute the equivalent euler triplet

        Arguments:
            no arguments

        Returns:
            euler_array: 3, numpy array of euler angles
        """
        euler_array = np.array([self.phi(), self.theta(), self.psi()])

        return euler_array

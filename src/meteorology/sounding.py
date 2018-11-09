#!/usr/bin/env python
import pdb

import numpy

import datetime
import geodesy.conversions

class Sounding(object):
    """ A class to deal with sounding data
    """
    def __init__(self, data=None):
        """ Constructor

        Arguments:
            data: a dictionary of all the data

        Returns:
            class instance
        """
        super(Sounding, self).__init__()
        self._P = None
        self._z = None
        self._T = None
        self._dew_point = None
        self._RH = None
        self._mixing_ratio = None
        self._wind_direction = None
        self._wind_speed = None
        self._theta = None
        self._theta_e = None
        self._theta_v = None

        self._time = None

        if data is not None:
            self.from_dict(data)

    def from_dict(self, data):
        """Populate sounding data from a dictionary of data

        Arguments:
            data: a dictionary of data with fields:
                P: pressure (Pa)
                z: altitude (m)
                T: tempeature (K)
                dew_point: dew point (K)
                self._RH: relative humidity (%)
                self._mixing_ratio = None
                self._wind_direction = None
                self._wind_speed = None
                self._theta = None
                self._theta_e = None
                self._theta_v = None

        Returns:
            no returns
        """
        self._P = numpy.array(data['P'])
        self._z = numpy.array(data['z'])
        self._T = numpy.array(data['T'])
        self._dew_point = numpy.array(data['dew_point'])
        self._u = numpy.array(data['u'])
        self._v = numpy.array(data['v'])

        sounding_datetime = datetime.datetime(
            data['year'],
            data['month'],
            data['day'],
            data['hour'],
            0,
            0,
            0)
        self._time = geodesy.conversions.datetime_to_gps(sounding_datetime)

        #From:
        #http://andrew.rsmas.miami.edu/bmcnoldy/Humidity.html
        self._RH = 100.0 * (
            numpy.exp(17.625 * (self._dew_point - 273.16) /
                (243.04 + self._dew_point - 273.16)) /
            numpy.exp(
                17.625 * (self._T - 273.16) / (243.04 + self._T - 273.16)))
        #From:
        #https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
        #and stull 13.1.4a
        e_ambient = 0.61078 * numpy.exp(
            17.2694 * (self._dew_point - 273.16) / (self._dew_point - 35.86))
        self._mixing_ratio = 621.97 * e_ambient / (self._P / 100.0 - e_ambient)
        self._theta = self._T * numpy.power(101325.0 / self._P, 0.286)
        # From stull table 13-1,
        # assuming L_v = 2450.0 and cp = 1005.7
        self._theta_e = self._theta + (
            2450.0 * self._theta / 1005.7 / self._T) * self._mixing_ratio
        # assumes no liquid water
        self._theta_v = self._theta * (1.0 + 0.61 * self._mixing_ratio)

        self._compute_gradients()

    def from_RAOB(self, data, timestamp):
        """ Populate sounding data from an RAOB formatted data (ie the format
        that rucsoundings and UWyo give soundings in)

        Arguments:
            data: nx11 numpy array giving sounding data
            timestamp: dict with fields
                year
                month
                day
                hour

        Returns:
            no returns
        """
        self._P = data[:,0] * 100.0
        self._z = data[:,1]
        self._T = data[:,2]
        self._dew_point = data[:,3]
        self._RH = data[:,4]
        self._mixing_ratio = data[:,5]
        self._u = -numpy.sin(numpy.deg2rad(data[:,6])) * data[:,7] * 0.51444
        self._v = -numpy.cos(numpy.deg2rad(data[:,6])) * data[:,7] * 0.51444
        self._theta = data[:,8]
        self._theta_e = data[:,9]
        self._theta_v = data[:,10]

        sounding_datetime = datetime.datetime(
            timestamp['year'],
            timestamp['month'],
            timestamp['day'],
            timestamp['hour'],
            0,
            0,
            0)
        self._time = geodesy.conversions.datetime_to_gps(sounding_datetime)

        self._compute_gradients()

    def _compute_gradients(self):
        """Compute the gradients of all parameters

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._d_P = numpy.gradient(self._P)
        self._d_z = numpy.gradient(self._z)
        self._d_T = numpy.gradient(self._T)
        self._d_dew_point = numpy.gradient(self._dew_point)
        self._d_RH = numpy.gradient(self._RH)
        self._d_mixing_ratio = numpy.gradient(self._mixing_ratio)
        self._d_u = numpy.gradient(self._u)
        self._d_v = numpy.gradient(self._v)
        self._d_theta = numpy.gradient(self._theta)
        self._d_theta_e = numpy.gradient(self._theta_e)
        self._d_theta_v = numpy.gradient(self._theta_v)

    @property
    def timestamp(self):
        """Get a timestamp for this sounding

        Arguments:
            no arguments

        Returns:
            timestamp: integer giving yyyymmddhh in utc
        """
        time_datetime = geodesy.conversions.gps_to_datetime(self._time)
        timestamp = (
            time_datetime.year * 1000000 +
            time_datetime.month * 10000 +
            time_datetime.day * 100 +
            imte_datetitme.hour)
        return timestamp

    @property
    def time(self):
        """Get the sounding time

        Arguments
            no arguments

        Returns:
            time: the time of the sounding as seconds since the GPS epoch
        """
        return self._time

    @property
    def hour(self):
        """Get the sounding hour

        Arguments:
            no arguments

        Returns:
            hour: the UTC hour of this sounding
        """
        time_datetime = geodesy.conversions.gps_to_datetime(self._time)
        return time_datetime.hour

    def P(self, z):
        """ Get the pressure at an altitude

        Arguments:
            z: the altitude of interest (m)

        Returns:
            P: the pressure at z (Pa)
        """
        return numpy.interp(z, self._z, self._P)

    def z(self, P):
        """ Get the altitude for a given pressure

        Arguments:
            P: the pressure level (Pa)

        Returns:
            z: the altitude (m)
        """
        return numpy.interp(P, self._P[::-1], self._z[::-1])

    def T(self, z=None, P=None):
        """ Get the temperature at altitude or pressure level

        Returns a tuple that gives temp at pressure and altitude. If one is not
        specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            Tz: temperature at altitudes (C)
            Tp: temperature at pressures (C)
        """
        return self._interp_at_levels(self._T, z, P)

    def dew_point(self, z=None, P=None):
        """ Get the dew point at an altitude or pressure level

        Returns a tuple that gives dew point at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            dew_point_z: dew point at altitudes (C)
            dew_point_P: dew point at pressures (C)
        """
        return self._interp_at_levels(self._dew_point, z, P)

    def relative_humidity(self, z=None, P=None):
        """ Get the relative_humidity at an altitude or pressure level

        Returns a tuple that gives RH at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            rh_z: relative humidity at altitudes (%)
            rh_p: relative humidity at pressures (%)
        """
        return self._interp_at_levels(self._RH, z, P)

    def mixing_ratio(self, z=None, P=None):
        """ Get the mixing ratio at an altitude or pressure level

        Returns a tuple that gives mixing ratio at pressure and altitude. If one
        is not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            mixing_ratio_z: mixing ratio at altitudes (g/kg)
            mixing_ratio_p: mixing ratio at pressures (g/kg)
        """
        return self._interp_at_levels(self._mixing_ratio, z, P)

    def wind_direction(self, z=None, P=None):
        """ Get the wind direction at an altitude or pressure level

        NOT CURRENTLY IMPLEMENTED (since I don't want to deal with the angle
        wrapping issue right now)

        Returns a tuple that gives wind direction at pressure and altitude. If
        one is not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            dir_z: wind direction at altitudes (rad)
            dir_p: wind direction at pressures (rad)
        """
        dir_z = None
        dir_p = None

        wind_z, wind_p = self.wind(z, P)

        if z is not None:
            dir_z = numpy.arctan2(-wind_z[:, 0], -wind_z[:, 1])
        if P is not None:
            dir_p = numpy.arctan2(-wind_z[:, 0], -wind_z[:, 1])
        return (dir_z, dir_p)

    def wind_speed(self, z=None, P=None):
        """ Get the wind speed at an altitude or pressure level

        Returns a tuple that gives wind speed at pressure and altitude. If one
        is not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            wind_speed_z: wind speed at altitudes (m/s)
            wind_speed_p: wind speed at pressures (m/s)
        """
        speed_z = None
        speed_p = None

        wind_z, wind_p = self.wind(z, P)

        if z is not None:
            speed_z = numpy.linalg.norm(numpy.array(wind_z, ndmin=2), axis=1)
        if P is not None:
            speed_p = numpy.linalg.norm(numpy.array(wind_p, ndmin=2), axis=1)
        return (speed_z, speed_p)

    def wind(self, z=None, P=None):
        """ Get the wind vector at an altitude or pressure level

        Returns a tuple that gives wind vector at pressure and altitude. If one
        is not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            wind_z: nx2 wind at altitudes (m/s)
                0: u component
                1: v component
            wind_p: nx2 wind at pressures (m/s)
                0: u component
                1: v component
        """
        wind_z = None
        wind_p = None

        u_z, u_p = self._interp_at_levels(self._u, z, P)
        v_z, v_p = self._interp_at_levels(self._v, z, P)

        if z is not None:
            wind_z = numpy.vstack((u_z, v_z)).T
        if P is not None:
            wind_p = numpy.vstack((u_p, v_p)).T
        return (wind_z, wind_p)

    def theta(self, z=None, P=None):
        """ Get the potential temperature at an altitude or pressure level

        Returns a tuple that gives theta at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            theta_z: potential temperature at altitudes (K)
            theta_p: potential temperature at pressures (K)
        """
        return self._interp_at_levels(self._theta, z, P)

    def theta_e(self, z=None, P=None):
        """ Get the equivalent potential temperature at an altitude or pressure
        level

        Returns a tuple that gives theta_e at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            theta_e_z: equivalent potential temperature at altitudes (K)
            theta_e_p: equivalent otential temperature at pressures (K)
        """
        return self._interp_at_levels(self._theta_e, z, P)

    def theta_v(self, z=None, P=None):
        """ Get the virtual potential temperature at altitude or pressure level

        Returns a tuple that gives theta_v at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: optional, the altitude of interest (m)
            P: optional, the pressure level (Pa)

        Returns:
            theta_v_z: virtual potential temperature at altitudes (K)
            theta_v_p: virtual potential temperature at pressures (K)
        """
        return self._interp_at_levels(self._theta_v, z, P)

    def P_gradient(self, z):
        """ Compute pressure gradient at altitudes

        Arguments:
            z: altitudes of interest

        Returns:
            dPdz: pressure gradient (Pa/m)
        """
        dP,_ = self._interp_at_levels(self._d_P, z)
        dz,_ = self._interp_at_levels(self._d_z, z)
        return dP/dz

    def height_gradient(self, P):
        """ Compute height gradient at pressures

        This is weird

        Arguments:
            P: pressure of interest

        Returns:
            dzdP: height gradient (m/Pa)
        """
        dP,_ = self._interp_at_levels(self._d_P, P=P)
        dz,_ = self._interp_at_levels(self._d_z, P=P)
        return dz/dP

    def temperature_gradient(self, z=None, P=None):
        """ Compute temperature gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dTdz: gradient wrt height at altitudes (C/m)
            dTdP: gradient wrt pressure at pressures (C/Pa)
        """
        return self._gradients_at_levels(self._d_T, z, P)

    def dew_point_gradient(self, z=None, P=None):
        """ Compute dew_point gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dDPdz: gradient wrt height at altitudes (C/m)
            dDPdP: gradient wrt pressure at pressures (C/Pa)
        """
        return self._gradients_at_levels(self._d_dew_point, z, P)

    def rh_gradient(self, z=None, P=None):
        """ Compute rh gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dRHdz: gradient wrt height at altitudes (%/m)
            dRHdP: gradient wrt pressure at pressures (%/Pa)
        """
        return self._gradients_at_levels(self._d_T, z, P)

    def mixing_ratio_gradient(self, z=None, P=None):
        """ Compute mixing ratio gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            drdz: gradient wrt height at altitudes (g/kg/m)
            drdP: gradient wrt pressure at pressures (g/kg/Pa)
        """
        return self._gradients_at_levels(self._d_mixing_ratio, z, P)

    def wind_gradient(self, z=None, P=None):
        """ Compute wind gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dVdz: nx2 gradient wrt height at altitudes (m/s/m)
            dVdP: nx2 gradient wrt pressure at pressures (m/s/Pa)
        """
        dVdz = None
        dVdp = None

        dudz, dudp = self._gradients_at_levels(self._d_u, z, P)
        dvdz, dvdp = self._gradients_at_levels(self._d_v, z, P)

        if z is not None:
            dVdz = numpy.vstack((dudz, dvdz)).T
        if P is not None:
            dVdp = numpy.vstack((dudp, dvdp)).T
        return (dVdz, dVdp)

    def theta_gradient(self, z=None, P=None):
        """ Compute potential temperature gradient at altitude or pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dthetadz: gradient wrt height at altitudes (K/m)
            dthetadP: gradient wrt pressure at pressures (K/Pa)
        """
        return self._gradients_at_levels(self._d_theta, z, P)

    def theta_e_gradient(self, z=None, P=None):
        """ Compute equivalent potential temperature gradient at altitude or
        pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dthetaedz: gradient wrt height at altitudes (C/m)
            dthetaedP: gradient wrt pressure at pressures (C/Pa)
        """
        return self._gradients_at_levels(self._d_theta_e, z, P)

    def theta_v_gradient(self, z=None, P=None):
        """ Compute virtual potential temperature gradient at altitude or
        pressures

        Returns a tuple that gives gradient at pressure and altitude. If one is
        not specified then its place in the tuple is None

        Arguments:
            z: altitudes of interest (m)
            P: pressures of interest (Pa)

        Returns:
            dthetavdz: gradient wrt height at altitudes (K/m)
            dthetavdP: gradient wrt pressure at pressures (K/Pa)
        """
        return self._gradients_at_levels(self._d_theta_v, z, P)

    def _interp_at_levels(self, val, z=None, P=None):
        """ Helper to do the z/P interpolation so I can just call this for
        whatever value is of interest...

        Arguments:
            val: the item to be interpolated
            z: altitude of interest
            P: pressure of interest

        Returns:
            val_z: value at altitudes
            val_p: value at pressures
        """
        val_z = None
        val_p = None

        if z is not None:
            val_z = numpy.interp(z, self._z, val)
        if P is not None:
            val_p = numpy.interp(P, self._P[::-1], val[::-1])

        return (val_z, val_p)

    def _gradients_at_levels(self, val, z=None, P=None):
        """ Helper to to z/P interpolation and compute gradients so I can just
        call this for whatever value

        Arguments:
            val: basic gradient of value of interest
            z: altitude of interest
            P: pressure of interest

        Returns:
            dval_dz: gradient wrt z at altitudes
            dval_dp: gradient wrt P at pressures
        """
        dval_dz = None
        dval_dp = None

        dval_z, dval_p = self._interp_at_levels(val, z, P)
        if z is not None:
            dz = numpy.interp(z, self._z, self._d_z)
            dval_dz = dval_z / dz
        if P is not None:
            dp = numpy.interp(P, self._P[::-1], self._d_P[::-1])
            dval_dp = dval_p / dp

        return (dval_dz, dval_dp)


# -*- coding: utf-8 -*-
import numpy

import astropy.time
import datetime

import numpy as np
from geometry.rotations import rotate
from geometry.conversions import d2r
from math import sin, cos, atan2, sqrt, pi
import warnings

def kml2lla(lla_kml):
    """ Alias to kml_to_lla
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' kml_to_lla from the same module.', DeprecationWarning)
    return kml_to_lla(lla_kml)

def kml_to_lla(lla_kml):
    """ long/lat/alt to lat/long/alt

    Converts long/lat/alt, ie from a KML file to lat/long/alt used by
    everyone else in the world also does the degrees to radians thing

    Arguments:
        lla_kml: nx3 numpy array giving long/lat/alt position in degrees and
            meters

    Returns:
        lla: nx3 numpy array giving lat/long/alt in rad/m
    """
    if lla_kml.ndim > 1:
        lla = np.zeros(lla_kml.shape)
        for (kml, i) in zip(lla_kml, range(0, lla_kml.shape[0])):
            lla[i] = kml2lla(kml)
        return lla

    lla = np.zeros(lla_kml.shape)
    lla[[0,1]] = d2r(lla_kml[[1,0]])
    lla[2] = lla_kml[2]
    return lla


def lla2ned(lla, lla_ref):
    """ Alias for lla_to_ned, which is what we're migraing to for consistency
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' lla_to_ned from the same module.', DeprecationWarning)
    return lla_to_ned(lla, lla_ref)

def lla_to_ned(lla, lla_ref):
    """ lla to ned

    Converts lat/long/alt to north/east/down

    Args:
        lla: nx3 numpy array giving lat/long/alt position in radians and meters
        lla_ref: 1x3 numpy array giving the lat/long/alt position to
            report ned points relative to

    Returns:
        ned: nx3 numpy array giving north/east/down positions

    Notes: all measured to WGS84
    """
    xyz = lla_to_xyz(lla)
    return xyz_to_ned(xyz, lla_ref)

def lla2xyz(lla):
    """ Alias to lla_to_xyz
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' lla_to_xyz from the same module.', DeprecationWarning)
    return lla_to_xyz(lla)

def lla_to_xyz(lla):
    """  lla to xyz

    converts lat/long/alt to ecef xyz

    Args:
        lla: lat/long/alt position in radians and meters

    Returns:
        xyz: ecef xyz position in meters

    Notes: all measured to WGS84
    """
    if lla.ndim > 1:
        xyz = np.zeros(lla.shape)
        for (coord,i) in zip(lla, range(lla.shape[0])):
            xyz[i] = lla2xyz(coord)
        return xyz

    a_earth = 6378137
    e2 = 0.006694379990140
    r_n = a_earth/sqrt(1 - e2*pow(sin(lla[0]),2))

    x = (r_n + lla[2]) * cos(lla[0]) * cos(lla[1])
    y = (r_n + lla[2]) * cos(lla[0]) * sin(lla[1])
    z = (r_n * (1 - e2) + lla[2]) * sin(lla[0])
    xyz = np.array([x,y,z])
    return xyz

def xyz2ned(xyz, lla_ref):
    """ Alias to xyz_to_ned
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' xyz_to_ned from the same module.', DeprecationWarning)
    return xyz_to_ned(xyz, lla_ref)

def xyz_to_ned(xyz, lla_ref):
    """ ecef xyz to ned

    converts ecef xyz coordinates to surface north/east/down

    Args:
        xyz: 3, or nx3 numpy array or giving locations in meters
        lla_ref: 1x3 numpy array giving the reference position to report
            north/east/down position relative to, radians and metersr

    Returns:
        ned: nx3 numpy array giving position in meters

    Notes: all measured to WGS84
    """
    if xyz.ndim > 1:
        ned = np.zeros(xyz.shape)
        for (coord, i) in zip(xyz, range(xyz.shape[0])):
            ned[i] = xyz2ned(coord, lla_ref)
        return ned

    xyz_ref = lla2xyz(lla_ref)

    R1 = rotate('z', pi/2 + lla_ref[0][1])
    R2 = rotate('x', pi/2 - lla_ref[0][0])
    R = np.dot(R2,R1)

    diff_xyz = xyz - xyz_ref
    enu = np.dot(R, diff_xyz.T).T
    return enu2ned(enu)

def ned2xyz(ned, lla_ref):
    """ Alias to ned2xyz
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' ned_to_xyz from the same module.', DeprecationWarning)
    return ned_to_xyz(ned, lla_ref)

def ned_to_xyz(ned, lla_ref):
    """ north/east/down to ecef xyz

    takes north/east/down coordinates and gives earth centered earth
    fixed xyz points

    Args:
        ned: 3, or nx3 numpy array them, north/east/down position in
            meters
        lla_ref: 1x3 numpy array giving lat/long/alt point ned is
            measured from in rad and meters

    Returns:
        xyz: nx3 numpy array giving xyz in m

    Notes: all measured to WGS84
    """
    if ned.ndim > 1:
        xyz = np.zeros(ned.shape)
        for (coords, i) in zip(ned, range(ned.shape[0])):
            xyz[i] = ned2xyz(coords, lla_ref)
        return xyz

    R1 = rotate('z', pi/2 + lla_ref[0][1])
    R2 = rotate('x', pi/2 - lla_ref[0][0])
    R = np.dot(R2,R1)

    enu = ned2enu(ned)

    xyz_ref = lla2xyz(lla_ref)

    diff_xyz = np.dot(R.T, enu.T).T
    return diff_xyz + xyz_ref

def xyz2lla(xyz):
    """ Alias to xyz_to_lla
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' xyz_to_lla from the same module.', DeprecationWarning)
    return xyz_to_lla(xyz)

def xyz_to_lla(xyz):
    """ ecef xyz to lla

    takes earth-fixed and centered xyz coordinates to lat/lon/alt

    Args:
        xyz: 3, or nx3 numpy array giving xyz in m

    Returns:
        lla: 3, or nx3 numpy array giving lat/lon/alt position in
            radians and meters

    Notes: all measured to WGS84
    """
    if xyz.ndim > 1:
        lla = np.zeros(xyz.shape)
        for (coords, i) in zip(xyz, range(xyz.shape[0])):
            lla[i] = xyz2lla(coords)
        return lla

    a_earth = 6378137.0
    flattening = 1/298.257223563
    e2 = (2 - flattening) * flattening

    if (xyz[0] == 0.0) and (xyz[1] == 0.0):
        lon = 0.0
    else:
        lon = atan2(xyz[1], xyz[0])

    if (np.sum(xyz) == 0.0):
        lla = np.zeros((3,))
    else:
        rho_sqr = xyz[0]*xyz[0] + xyz[1]*xyz[1]
        rho = sqrt(rho_sqr)

        lat_temp = atan2(xyz[2],rho)
        alt_temp = sqrt(rho_sqr + xyz[2]*xyz[2]) - a_earth

        rho_error = 10000.0
        alt_error = 10000.0

        niter = 0
        while (((abs(rho_error) > 1e-6) or (abs(alt_error) > 1e-6))
            and (niter < 1e4)):
            niter = niter + 1
            sin_lat = sin(lat_temp)
            cos_lat = cos(lat_temp)

            q = 1.0 - e2 * sin_lat * sin_lat
            r_n = a_earth / sqrt(q)
            drdl = r_n * e2 * sin_lat * cos_lat / q

            rho_error = (r_n + alt_temp) * cos_lat - rho
            alt_error = (r_n *(1 - e2) + alt_temp) * sin_lat - xyz[2]

            aa = drdl * cos_lat - (r_n + alt_temp) * sin_lat
            bb = cos_lat
            cc = (1 - e2) * (drdl * sin_lat + r_n * cos_lat)
            dd = sin_lat

            det_inv = 1.0/(aa*dd - bb*cc)
            lat_temp = lat_temp - det_inv * (dd * rho_error -
                bb * alt_error)
            alt_temp = alt_temp - det_inv * (-cc * rho_error +
                aa * alt_error)

            if (niter == 1e4):
                N = a_earth
                Nph = sqrt(np.sum(np.multiply(xyz, xyz)))
                lat = 0
                lat_last = 100
                niter = 0
                while ((abs(lat - lat_last) > 1e-6) and (niter < 1e4)):
                    niter = niter + 1
                    lat_last = lat

                    lat = atan2(xyz[2], sqrt(xyz[0] * xyz[0] +
                        xyz[1] * xyz[1]) * (1 - N * e2 / Nph))
                    N = a_earth / sqrt(1 - e2 * sin(lat) * sin(lat))
                    Nph = sqrt(xyz[0] * xyz[0] + xyz[1] *
                        xyz[1]) / cos(lat)

                    h = Nph - N

                    alt_temp = h
                    lat_temp = lat

    return np.array([lat_temp, lon, alt_temp])

def ned2lla(ned, lla_ref):
    """ Alias to ned_to_lla
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' ned_to_lla from the same module.', DeprecationWarning)
    return ned_to_lla(ned, lla_ref)

def ned_to_lla(ned, lla_ref):
    """ converts from north/east/down to lat/lon/alt

    Converts north/east/down meters to lat/lon/alt radians and meters

    Args:
        ned: 3, or nx3 numpy array giving north/east/down position in
            meters
        lla_ref: 3, numpy array giving lat/lon/alt reference point in
            radians and meters that ned is measured from

    Returns:
        lla: lat/lon/alt position in radians and meters

    Notes: all measured to WGS84
    """
    xyz = ned2xyz(ned, lla_ref)
    lla = xyz2lla(xyz)
    return lla

def enu2ned(enu):
    """ Alias to enu_to_ned
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' enu_to_ned from the same module.', DeprecationWarning)
    return enu_to_ned(enu)

def enu_to_ned(enu):
    """
    convert from enu to ned

    takes an nx3 element numpy array defining a vector in east/north/up
    coordinates and converts to north/east/down

    Args:
        enu: 3, or nx3 numpy array

    Returns:
        ned: 3, nx3 numpy array matching input dimensions

    Notes: all measured to WGS84
    """
    if enu.ndim > 1:
        ned = np.zeros(enu.shape)
        for (coords, i) in zip(enu, range(enu.shape[0])):
            ned[i] = enu2ned(coords)
        return ned

    R = np.zeros([3,3])
    R[1,0] = 1.0
    R[0,1] = 1.0
    R[2,2] = -1.0

    return np.dot(R, enu)

def ned2enu(ned):
    """ Alias to ned_to_enu
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' ned_to_enu from the same module.', DeprecationWarning)
    return ned_to_enu(ned)

def ned_to_enu(ned):
    """
    convert from ned to enu

    takes an 3, or nx3 element numpy array defining a vector in
    north/east/down coordinates and converts to east/north/up

    Args:
        enu: 3, or nx3 numpy array

    Returns:
        ned: 3, or nx3 numpy array

    Notes: all measured to WGS84
    """
    if ned.ndim > 1:
        enu = np.zeros(ned.shape)
        for (coords, i) in zip(ned, range(ned.shape[0])):
            enu[i] = ned2enu(coords)
        return enu

    R = np.zeros([3,3])
    R[1,0] = 1.0
    R[0,1] = 1.0
    R[2,2] = -1.0

    return np.dot(R, ned)

def get_distance(lla0, lla1):
    """Method uses the speherical cosine law to return a distance. This is
        fast, but not particularly accurate. For more accuracy, use the
        Haversine formula: http://www.movable-type.co.uk/scripts/latlong.html

        Args:
            lla0: a list [lat, lon] with the origin
            lla1: a list [lat, lon] with the destintation

        Returns:
            distance between X0 and X1 in meters
        """
    lat0 = lla0[0]
    lon0 = lla0[1]
    lat1 = lla1[0]
    lon1 = lla1[1]
    delta_lon = lon1 - lon0
    R = 6371000.0
    return np.arccos(np.sin(lat1) * np.sin(lat0) +
                     np.cos(lat1) * np.cos(lat0) * np.cos(delta_lon)) * R

def datetime_to_gps(epoch):
    """ Convert a datetime instance to seconds since the gps epoch.

    Arguments:
        epoch: datetime instance specifying the epoch of interest. Can also
            be list or tuple of same.

    Returns:
        secs: floating point seconds since gps epoch. if input was list or
            tuple then will return a numpy array
    """
    if isinstance(epoch, tuple) or isinstance(epoch, list):
        return numpy.array([datetime_to_gps(e) for e in epoch])

    assert isinstance(epoch, datetime.datetime),\
        "epoch must be a datetime instance"
    ap_epoch = astropy.time.Time(epoch, scale='tai')
    secs = ap_epoch.gps
    return secs

def gps_to_datetime(secs):
    """ Convert seconds since the gps epoch to a datetime instance.

    Arguments:
        secs: floating point seconds since gps epoch. Can also be numpy
            ndarray, list, or tuple of same.

    Returns:
        epoch: datetime instance specifying the epoch of interest. if input
            was numpy array, list, or tuple then will return a tuple of
            datetime instances
    """
    if (isinstance(secs, numpy.ndarray) or
        isinstance(secs, list) or
        isinstance(secs, tuple)):
        return tuple(gps_to_datetime(s) for s in secs)

    assert isinstance(secs, float), "gps time must be float seconds"
    ap_epoch = astropy.time.Time(secs, scale='tai', format='gps')
    epoch = ap_epoch.datetime
    return epoch

def datetime_to_unix(epoch):
    """ Convert a datetime instance to seconds since the unix epoch.

    Arguments:
        epoch: datetime instance specifying the epoch of interest. can also
            be a list or tuple of datetime

    Returns:
        secs: floating point seconds since unix epoch. If input was list or
            tuple then will return a numpy array
    """
    if isinstance(epoch, tuple) or isinstance(epoch, list):
        return numpy.array([datetime_to_unix(e) for e in epoch])

    assert isinstance(epoch, datetime.datetime),\
        "epoch must be a datetime instance"
    ap_epoch = astropy.time.Time(epoch, scale='tai')
    secs = ap_epoch.unix
    return secs

def unix_to_datetime(secs):
    """ Convert seconds since the unix epoch to a datetime instance.

    Arguments:
        secs: floating point seconds since unix epoch. Can be a numpy array,
            list, or tuple as well

    Returns:
        epoch: datetime instance specifying the epoch of interest. If input was
            numpy array, list, or tuple then this will return a tuple of
            datetime instances
    """
    if (isinstance(secs, numpy.ndarray) or
        isinstance(secs, list) or
        isinstance(secs, tuple)):
        return tuple(unix_to_datetime(s) for s in secs)

    assert isinstance(secs, float), "unix time must be float seconds"
    ap_epoch = astropy.time.Time(secs, scale='tai', format='unix')
    epoch = ap_epoch.datetime
    return epoch


def gps_to_unix(gps_secs):
    """Convert seconds since the gps epoch to seconds since unix epoch

    Arguments:
        gps_secs: floating point seconds since gps epoch. Can also be numpy
            ndarray, list, or tuple of same.

    Returns:
        unix_secs: corresponding floating point seconds since unix epcoh. if
            input was numpy array, list, or tuple then will return a tuple of
            datetime instances
    """
    if (isinstance(gps_secs, numpy.ndarray) or
        isinstance(gps_secs, list) or
        isinstance(gps_secs, tuple)):
        return tuple(s.gps_to_unix for s in gps_secs)

    assert isinstance(gps_secs, float), "gps time must be float seconds"
    epoch = astropy.time.Time(gps_secs, scale='tai', format='gps')
    return epoch.unix

def unix_to_gps(unix_secs):
    """Convert seconds since the unix epoch to seconds since gps epoch

    Arguments:
        unix_secs: floating point seconds since unix epoch. Can also be numpy
            ndarray, list, or tuple of same.

    Returns:
        gps_secs: corresponding floating point seconds since gps epoch. if
            input was numpy array, list, or tuple then will return a tuple of
            datetime instances
    """
    if (isinstance(unix_secs, numpy.ndarray) or
        isinstance(unix_secs, list) or
        isinstance(unix_secs, tuple)):
        return tuple(s.gps_to_unix for s in unix_secs)

    assert isinstance(unix_secs, float), "unix time must be float seconds"
    epoch = astropy.time.Time(unix_secs, scale='tai', format='unix')
    return epoch.gps

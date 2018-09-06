
import pdb

import os

import re
import datetime

import numpy
import scipy.interpolate

import geodesy.conversions

import parsers.nmea

import time

class PerlanParser(parsers.nmea.NMEA):
    """Parser for data stream from the perlan

    The Perlan data stream is implemented in an nmea-like way so we'll inherit
    from the nmea parser and redefine as needed. Note the LXNAV messages do
    not include a time so we're going to implicitly pick it up from the GPSRMC
    messages. This means that the time in LXNAV messages could be off by up to
    the GPSRMC interval (usually 1 second)
    """
    def __init__(self, file_path=None, string_data=None):
        """constructor

        Arguments:
            file_path: path to a file with nmea data in it
            string_data: alternate option, can directly pass nmea data to
                be parsed.

        Returns:
            class instance
        """
        super(PerlanParser, self).__init__()

        self._latest_time = None

        self._time_lxwp0 = []
        self._baro_altitude = []
        self._v_ias = []
        self._edot = []
        self._psi = []
        self._wind = []

        self._time_therm = []
        self._OAT = []
        self._therm_field_1 = []
        self._therm_field_2 = []

        additional_parsers = {
            'LXWP0': self.parse_lxwp0,
            'therm': self.parse_therm,
            }
        self._sentence_parsers.update(additional_parsers)

        self._extract_sentence_header = self._get_perlan_header

        if file_path is not None:
            self.parse_file(file_path)
            return
        if string_data is not None:
            self.parse_string(string_data)
            return

    def _get_perlan_header(self, string_data):
        """Get an nmea header from string data

        Arguments:
            string_data: a string with an nmea sentence

        Returns:
            header: string with just the sentence id
        """
        search_result = re.search('.+\:\$([a-zA-Z0-9]+),', string_data)
        if not search_result:
            return
        return search_result.groups()[0]

    def parse_rmc(self, string_data, save=True):
        """Redefinition of parse_rmc to save the time
        """
        if ':' in string_data:
            string_data = string_data.split(':')[1]
        rmc_data = super(PerlanParser, self).parse_rmc(string_data, save)
        if rmc_data:
            self._latest_time = rmc_data[0]
        else:
            self._latest_time = None

    def parse_lxwp0(self, string_data, save=True):
        """Parse an lxnav LXWP0 message

        Arguments:
            string_data: string with an LXWP0 message in it
            save: save this data

        Returns:
            lxwp0_data: tuple containing lxwp0 data
                v_ias: indicated airspeed (m/s)
                h_baro: barometric altitude (m)
                edot: vario reading (m/s)
                psi: heading (rad)
                u_wind: east wind component (m/s)
                v_wind: north wind component (m/s)
        """
        data = re.split(',|\*', string_data)

        if len(data) <= 13:
            return None
        if len(self._time_lxwp0) > 0:
            if self._latest_time == self._time_lxwp0[-1]:
                return None

        v_ias = float(data[2]) * 1000.0 / 3600.0
        h_baro = float(data[3])
        edot = float(data[4])

        if data[10] == '':
            psi = numpy.nan
        else:
            psi = numpy.deg2rad(float(data[10]))

        if data[11] == '':
            wind_psi = numpy.nan
        else:
            wind_psi = numpy.deg2rad(float(data[11]))

        if data[12] == '':
            wind_M = numpy.nan
        else:
            wind_M  = float(data[12])

        u_wind = -wind_M * numpy.sin(wind_psi)
        v_wind = -wind_M * numpy.cos(wind_psi)

        if save and self._latest_time is not None:
            self._time_lxwp0.append(self._latest_time)

            self._baro_altitude.append(h_baro)
            self._v_ias.append(v_ias)
            self._edot.append(edot)
            self._psi.append(psi)
            self._wind.append(numpy.array([u_wind, v_wind]))

    def parse_therm(self, string_data, save=True):
        """Parse a perlan therm message

        Arguments:
            string_data: string with an LXWP0 message in it
            save: save this data

        Returns:
            lxwp0_data: tuple containing lxwp0 data
                v_ias: indicated airspeed (m/s)
                h_baro: barometric altitude (m)
                edot: vario reading (m/s)
                psi: heading (rad)
                u_wind: east wind component (m/s)
                v_wind: north wind component (m/s)
        """
        data = re.split(',|\*', string_data)

        if len(data) <= 3:
            return None

        if len(self._time_therm) > 0:
            if self._latest_time == self._time_therm[-1]:
                return None

        OAT = float(data[1])
        therm_field_1 = float(data[2])
        therm_field_2 = float(data[3])

        if save and self._latest_time is not None:
            self._time_therm.append(self._latest_time)

            self._OAT.append(OAT)
            self._therm_field_1.append(therm_field_1)
            self._therm_field_2.append(therm_field_2)

    def baro_altitude(self, time=None):
        """ Get barometric altitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            baro_altitude: in m at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_lxwp0),
                numpy.array(self._baro_altitude))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_baro_altitude(time))

    def v_ias(self, time=None):
        """ Get indicated airspeed at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            v_ias: in m/s at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_lxwp0),
                numpy.array(self._v_ias))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_v_ias(time))

    def edot(self, time=None):
        """ Get vario reading at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            edot: specific total energy rate in m/s at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_lxwp0),
                numpy.array(self._edot))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_edot(time))

    def psi(self, time=None):
        """ Get heading at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            psi: heading angle in radians at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_lxwp0),
                numpy.array(self._psi))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_psi(time))

    def wind(self, time=None):
        """ Get wind at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            wind: wind vector in m/s at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_lxwp0),
                numpy.array(self._wind))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_wind(time))

    def OAT(self, time=None):
        """Get outside air temperature at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            OAT: outside air temperature at requested times
        """
        if time is None:
            return (
                numpy.array(self._time_therm),
                numpy.array(self._OAT))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_OAT(time))

    def therm_field_1(self, time=None):
        """Get therm message field 1

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            value: values of therm_field_1 at requested times
        """
        if time is None:
            return (
                numpy.array(self._time_therm),
                numpy.array(self._therm_field_1))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_therm_field_1(time))

    def therm_field_2(self, time=None):
        """Get therm message field 1

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            value: values of therm_field_2 at requested times
        """
        if time is None:
            return (
                numpy.array(self._time_therm),
                numpy.array(self._therm_field_2))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_therm_field_2(time))

    def clear_interp(self):
        """Clear interpolators so we can pickle
        """
        super(PerlanParser, self).clear_interp()

        self._interp_baro_altitude = None
        self._interp_v_ias = None
        self._interp_edot = None
        self._interp_psi = None
        self._interp_wind = None

        self._interp_OAT = None
        self._interp_therm_field_1 = None
        self._interp_therm_field_2 = None

    def _generate_interps(self):
        """Generate interpolating functions

        Arguments:
            no arguments

        Returns:
            no returns
        """
        while self._reading:
            time.sleep(0.001)

        self._interp_baro_altitude = scipy.interpolate.interp1d(
            numpy.array(self._time_lxwp0),
            numpy.array(self._baro_altitude))
        self._interp_v_ias = scipy.interpolate.interp1d(
            numpy.array(self._time_lxwp0), numpy.array(self._v_ias))
        self._interp_edot = scipy.interpolate.interp1d(
            numpy.array(self._time_lxwp0), numpy.array(self._edot))
        self._interp_psi = scipy.interpolate.interp1d(
            numpy.array(self._time_lxwp0), numpy.array(self._psi))
        self._interp_wind= scipy.interpolate.interp1d(
            numpy.array(self._time_lxwp0), numpy.array(self._wind), axis=0)
        self._interp_OAT= scipy.interpolate.interp1d(
            numpy.array(self._time_therm), numpy.array(self._OAT), axis=0)
        self._interp_therm_field_1 = scipy.interpolate.interp1d(
            numpy.array(self._time_therm),
            numpy.array(self._therm_field_1), axis=0)
        self._interp_therm_field_2 = scipy.interpolate.interp1d(
            numpy.array(self._time_therm),
            numpy.array(self._therm_field_2), axis=0)

        super(PerlanParser, self)._generate_interps()

import pdb

import os

import re
import datetime

import numpy
import scipy.interpolate

import geodesy.conversions

class IGC(object):
    """ A class for representing and manipulating flight data from an IGC file
    """
    def __init__(self, fname=None):
        """ Constructor

        Arguments:
            fname: optionally, the file name to build this flight from. If not
                specified, then internals will be initialized but left empty

        Returns:
            class instance
        """
        self._set_init_state()

        if fname is not None:
            self.from_igc_file(fname)
            return

    def _set_init_state(self):
        """ Clear all fields to their initial state

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._date = None
        self.pilot = None
        self.crew = None
        self.glider = None
        self.registration = None
        self.datum = None
        self.contest_id = None

        self._time = []
        self._latitude = []
        self._longitude = []
        self._pressure_altitude = []
        self._gps_altitude = []
        self._valid = []

        self.clear_interp()

    def from_igc_file(self, fname):
        """ Populate data for this flight from an IGC file

        Arguments:
            fname: file path/name to the igc file to be used

        Returns:
            no returns
        """
        assert os.path.isfile(fname), 'invalid filepath specified'

        self._set_init_state()

        igc_file = open(fname, 'r')

        record_parsers = {
            'A': self._parse_a, 'B': self._parse_b, 'C': self._parse_c,
            'D': self._parse_d, 'E': self._parse_e, 'F': self._parse_f,
            'G': self._parse_g, 'H': self._parse_h, 'I': self._parse_i,
            'J': self._parse_j, 'K': self._parse_k, 'L': self._parse_l}

        # clear out old flight data before reading in a new flight


        # IGC files are a series of one-line records with the first letter
        # identifying the record type. Make sure we have a parser for that
        # type then feed the line to it.
        for line in igc_file.readlines():
            if line[0] in record_parsers:
                record_parsers[line[0]](line)


    def _parse_a(self, line):
        """ Parse the A (FR ID number) record

        ignores "ID extension"

        Arguments:
            line: the line from the log file

        Returns:
            no returns
        """
        self._manufacturer = line[1:4]
        self._serial = line[4:7]

    def _parse_b(self, line):
        """ Parse the B record (GPS fix)

        Arguments:
            line: the line from the log file

        Returns:
            no returns
        """
        fix_time = datetime.datetime(
            self._last_epoch.year,
            self._last_epoch.month,
            self._last_epoch.day,
            int(line[1:3]), int(line[3:5]), int(line[5:7]))
        dt = fix_time - self._last_epoch
        if dt.days < 0:
            fix_time += datetime.timedelta(1)
        self._last_epoch = fix_time
        self._time.append(geodesy.conversions.datetime_to_gps(fix_time))

        hemisphere = {'N': 1.0, 'S': -1.0, 'E': 1.0, 'W': -1.0}

        latitude = numpy.deg2rad(
            float(line[7:9]) + float(line[9:14])/60000.0) * hemisphere[line[14]]
        self._latitude.append(latitude)

        longitude = numpy.deg2rad(
            float(line[15:18]) + float(line[18:23])/60000.0)
        longitude *= hemisphere[line[23]]
        self._longitude.append(longitude)

        valid = float(line[24] == 'V' or line[24] == 'A')
        self._valid.append(valid)

        p_alt = float(line[25:30])
        self._pressure_altitude.append(p_alt)
        gps_alt = float(line[30:35])
        self._gps_altitude.append(gps_alt)

        # do things to handle additional data, but worry about it later

    def _parse_c(self, line):
        """ Parse the C record (task declaration)

        Not implemented yet
        """
        return

    def _parse_d(self, line):
        """ Parse the D record (differential GPS)

        Not implemented yet
        """

    def _parse_e(self, line):
        """ Parse the E record (events)

        Not implemented yet
        """

    def _parse_f(self, line):
        """ Parse the F record (satellite constellation)

        Not implemented yet
        """

    def _parse_g(self, line):
        """ Parse the G record (security)

        Not implemented yet
        """

    def _parse_h(self, line):
        """ Parse the H record (file header)

        Arguments:
            line: line from the log file to parse

        Returns:
            no returns
        """
        header_dict = {
            'HFDTE': self._set_date,
            'HFPLT': lambda line: self._set_field('pilot', line),
            'HFCM2': lambda line: self._set_field('crew', line),
            'HFGTY': lambda line: self._set_field('glider', line),
            'HFGID': lambda line: self._set_field('registration', line),
            'HFDTM': lambda line: self._set_field('datum', line),
            'HFCID': lambda line: self._set_field('contest_id', line),
            }

        if line[0:5] not in header_dict:
            return

        header_dict[line[0:5]](line)

    def _set_date(self, line):
        """ Set the start date of the file

        Arguments:
            line: line from the igc file to parse

        Returns:
            no returns
        """
        day = int(line[5:7])
        month = int(line[7:9])
        year = int(line[9:11]) + 2000
        self._last_epoch = datetime.datetime(year, month, day, 0, 0, 0)
        self._date = datetime.date(year, month, day)
        return

    def _set_field(self, field, line):
        """ Set a field in the class from a header entry

        Arguments:
            field: the name of the field to set
            line: line from the igc file to parse

        Returns:
            no returns
        """
        # everything after the colon is data entry
        entry = re.search('.+:(.+)', line)

        if entry is None:
            return

        setattr(self, field, entry.groups()[0][0:-1])

    def _parse_i(self, line):
        """ Parse I record (B record extension)

        Not yet implemented
        """

    def _parse_j(self, line):
        """ Parse J record (K record extension)

        Not yet implemented
        """

    def _parse_k(self, line):
        """ Parse K record (lower rate information)

        Not yet implemented
        """

    def _parse_l(self, line):
        """ Parse L record (log book/comments)

        Not yet implemented
        """

    @property
    def time(self):
        """ getter for time
        """
        return numpy.array(self._time)

    def lla(self, time=None):
        """ Get lat/long/alt at specified times

        Alt here is gps alt

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd

        Returns:
            lla: numpy array of lla
        """
        return numpy.vstack((
            self.latitude(time),
            self.longitude(time),
            self.gps_altitude(time))).T

    def latitude(self, time=None):
        """ Get latitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd

        Returns:
            latitude: in radians at specified epochs
        """
        if time is None:
            return numpy.array(self._latitude)

        if not self._is_interps_current:
            self._generate_interps()

        return self._interp_latitude(time)

    def longitude(self, time=None):
        """ Get longitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd

        Returns:
            longitude: in radians at specified epochs
        """
        if time is None:
            return numpy.array(self._longitude)

        if not self._is_interps_current:
            self._generate_interps()

        return self.interp_longitude(time)

    def gps_altitude(self, time=None):
        """ Get gps altitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd

        Returns:
            gps_altitude: in meters at specified epochs
        """
        if time is None:
            return numpy.array(self._gps_altitude)

        if not self._is_interps_current:
            self._generate_interps()

        return self._interp_gps_altitude(time)

    def pressure_altitude(self, time=None):
        """ Get pressure altitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd

        Returns:
            pressure_altitude: in meters at specified epochs
        """
        if time is None:
            return numpy.array(self._pressure_altitude)

        if not self._is_interps_current:
            self._generate_interps()

        return self._interp_pressure_altitude(time)

    def _generate_interps(self):
        """ Generate interpolating functions

        Arguments:
            no arguments

        Returns:
            no returns
        """
        t = numpy.array(self._time)

        self._interp_latitude = scipy.interpolate.interp1d(
            t, numpy.array(self._latitude))
        self.interp_longitude = scipy.interpolate.interp1d(
            t, numpy.array(self._longitude))
        self._interp_pressure_altitude = scipy.interpolate.interp1d(
            t, numpy.array(self._pressure_altitude))
        self._interp_gps_altitude = scipy.interpolate.interp1d(
            t, numpy.array(self._gps_altitude))

        self._is_interps_current = True

    def clear_interp(self):
        """ Clear interpolating functions so that this can be pickled

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._interp_latitude = None
        self._interp_longitude = None
        self._interp_pressure_altitude = None
        self._interp_gps_altitude = None
        self._is_interps_current = False

    @property
    def start_year(self):
        """Get the year the flight started in

        Returns:
            year: the year that the flight started in, derived from the IGC
                HFDTE entry
        """
        return self._date.year

    @property
    def start_month(self):
        """Get the month the flight srated in

        Returns:
            month: the month the flight started in, reported by the HFDTE entry
        """
        return self._date.month

    @property
    def start_day(self):
        """Get the day the flight started on

        Returns:
            day: the day the flight started on, reported by the HFDTE entry
        """
        return self._date.day

    @property
    def start_date(self):
        """Get the datetime instance identifying the start date

        Returns:
            start: datetime entry specifying the start date reported by the
                HFDTE entry in the igc file
        """
        return self._date

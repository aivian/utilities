import pdb

import os

import re
import datetime

import numpy
import scipy.interpolate

import geodesy.conversions

class NMEA(object):
    """Parser for NMEA data
    """
    def __init__(self, file_path=None, string_data=None):
        """Constructor

        Arguments:
            file_path: path to a file with nmea data in it
            string_data: alternate option, can directly pass nmea data to
                be parsed.

        Returns:
            class instance
        """
        self._reading = False

        self._time_latitude = []
        self._latitude = []
        self._time_longitude = []
        self._longitude = []
        self._time_speed = []
        self._ground_speed = []
        self._time_track = []
        self._ground_track = []

        self._sentence_parsers = {
            'GPRMC': self.parse_rmc,
            }

        self._extract_sentence_header = self._get_nmea_header

        self.clear_interp()

        if file_path is not None:
            self.parse_file(file_path)
            return
        if string_data is not None:
            self.parse_string(string_data)
            return

    def clear_interp(self):
        """Clear interpolators so that this can be pickled

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._is_interps_current = False

        self._interp_latitude = None
        self._interp_longitude = None
        self._interp_ground_speed = None
        self._interp_track = None

    def parse_file(self, file_path):
        """Parse a file of nmea data

        Arguments:
            file_path: path to file to parse

        Returns:
            no returns
        """
        self._reading = True
        with open(file_path, 'r') as nmea_file:
            for line in nmea_file.readlines():
                self.parse_sentence(line, save=True)
        self._reading = False

    def parse_string(self, string_data):
        """Parse a bunch of string data

        Arguments:
            string_data: can either be a tuple or list of strings that have
                already been separated by new lies, or else a big string with
                the newline characters still in there.

        Returns:
            no returns
        """
        if isinstance(string_data, basestring):
            string_data = string_data.split('\n')
        for line in string_data:
            self.parse_sentence(line, save=True)

    def _get_nmea_header(self, string_data):
        """Get an nmea header from string data

        Arguments:
            string_data: a string with an nmea sentence

        Returns:
            header: string with just the sentence id
        """
        search_result = re.search('\$([A-Z0-9]+),', string_data)
        if not search_result:
            return
        return search_result.groups()[0]

    def parse_sentence(self, string_data, save=True):
        """Parse an NMEA sentence into its parts

        Arguments:
            string_data: string containing an nmea sentence
            save: optional boolean, defaults True. If set True then save this
                sentence data

        Returns:
            sentence_data:
                id: string identifying this sentence
                data: tuple of data from this sentence
        """
        sentence = self._extract_sentence_header(string_data)
        if sentence not in self._sentence_parsers:
            return ('', tuple())
        sentence_data = self._sentence_parsers[sentence](string_data, save=True)
        # we have new data so clear our interpolators if we're saving this
        if save:
            self.clear_interp()
        return (sentence, sentence_data)

    def parse_rmc(self, string_data, save=True):
        """Parse an rmc sentence

        Arguments:
            string_data: string containing an RMC sentence
            save: boolean, defaults True. Save this data to the object

        Returns:
            rmc_data: tuple containing rmc data. returns None if invalid
                time: time of this data (gps seconds)
                latitude: latitude (radians)
                longitude: (radians)
                speed: ground speed (m/s)
                course: course track (radians)
        """
        def dm_to_sd(dm):
            """Convert degrees / minutes to decimal degrees

            Arguments:
                dm: string containing degrees and decimal minutes
                    concatenated togeter. Ex:
                    "12319.943281" = 123 degrees, 19.953281 minutes)

            Returns:
                dd: decimal degrees (floating point)
             """
            if not dm or dm == '0':
                return 0.
            d, m = re.match(r'^(\d+)(\d\d\.\d+)$', dm).groups()
            return float(d) + float(m) / 60

        data = string_data.split(',')
        if len(data) < 10:
            return None
        if (
            data[0] != '$GPRMC' or
            data[2] == 'V' or not
            self.verify_checksum(string_data)):
            return None
        year = int(data[9][4:6]) + 2000
        month = int(data[9][2:4])
        day = int(data[9][0:2])
        hour = int(data[1][0:2])
        minute = int(data[1][2:4])
        second = float(data[1][4:])
        n_s = 2 * (data[4] == 'N') - 1
        e_w = 2 * (data[6] == 'E') - 1
        latitude = numpy.deg2rad((dm_to_sd(data[3]))  * n_s)
        longitude = numpy.deg2rad((dm_to_sd(data[5])) * e_w)
        speed = float(data[7]) * 0.5144 # knots to m/s
        course = numpy.deg2rad(float(data[8]))

        time = geodesy.conversions.datetime_to_gps(
            datetime.datetime(
                year,
                month,
                day,
                hour,
                minute,
                int(second),
                int((second - int(second)) * 1000)))

        if save:
            self._time_latitude.append(time)
            self._latitude.append(latitude)
            self._time_longitude.append(time)
            self._longitude.append(longitude)
            self._time_speed.append(time)
            self._ground_speed.append(speed)
            self._time_track.append(time)
            self._ground_track.append(course)

        return (time, latitude, longitude, speed, course)

    def latitude(self, time=None):
        """ Get latitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            latitude: in radians at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_latitude), numpy.array(self._latitude))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_latitude(time))

    def longitude(self, time=None):
        """ Get longitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            longitude : in radians at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_longitude), numpy.array(self._longitude))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_longitude(time))

    def ground_speed(self, time=None):
        """ Get ground speed at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            ground_speed: in m/s at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_longitude), numpy.array(self._longitude))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_longitude(time))

    def longitude(self, time=None):
        """ Get longitude at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            longitude : in radians at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_longitude), numpy.array(self._longitude))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_longitude(time))

    def ground_speed(self, time=None):
        """ Get ground speed at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            ground_speed: in m/s at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_speed), numpy.array(self._ground_speed))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_ground_speed(time))

    def ground_track(self, time=None):
        """ Get ground track at specified times

        Arguments:
            time: optionally, the epochs of interest. if not specified all
                epochs are returnd. time in GPS seconds

        Returns:
            time: time values we got the latitude at
            ground_track: in radians at specified epochs
        """
        if time is None:
            return (
                numpy.array(self._time_track), numpy.array(self._ground_track))

        if not self._is_interps_current:
            self._generate_interps()

        return (time, self._interp_track(time))

    def _generate_interps(self):
        """Generate interpolating functions

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._interp_latitude = scipy.interpolate.interp1d(
            numpy.array(self._time_latitude), numpy.array(self._latitude))
        self._interp_longitude = scipy.interpolate.interp1d(
            numpy.array(self._time_longitude), numpy.array(self._longitude))
        self._interp_ground_speed = scipy.interpolate.interp1d(
            numpy.array(self._time_speed), numpy.array(self._ground_speed))
        self._interp_track = scipy.interpolate.interp1d(
            numpy.array(self._time_track), numpy.array(self._ground_track))

        self._is_interps_current = True

    def verify_checksum(self, string_data):
        """Verify the checksum in a nmea packet

        Arguments:
            string_data: a string containing exactly one nmea packet
                it must not contain $ or * anywhere except at the beginning of
                the packet and beginning of the checksum respectively

        Returns:
            good_packet: True if the checksum is correct. False otherwise
        """
        find_checksum = re.search('\*([A-Z0-9]{2})', string_data)
        if not find_checksum:
            return False
        reported_checksum = find_checksum.groups()[0]

        packet_interior = re.search('\$(.+)\*', string_data)
        if not packet_interior:
            return False
        computed_checksum = 0
        for byte in packet_interior.groups()[0]:
            computed_checksum ^= ord(byte)

        return hex(computed_checksum)[2:] == reported_checksum

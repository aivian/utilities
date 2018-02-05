import pdb

import os
import re
import urllib2

import datetime

import numpy

import geodesy.conversions

import meteorology.sounding

class GSDParser(object):
    """parse and hold a GSD format sounding
    """
    def __init__(self):
        """Constructor

        Arguments:
            None

        Returns:
            class isntance
        """
        self._parser = {
            1: self._parse_station_info,
            2: self._parse_checks,
            3: self._parse_station_id,
            4: self._parse_level,
            5: self._parse_level,
            6: self._parse_level,
            7: self._parse_level,
            8: self._parse_level,
            9: self._parse_level,
            }

        self._month_dict = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    def from_file(self, file_path):
        """Parse a sounding out of a file

        Arguments:
            file_path: a path to a saved file

        Returns:
            sounding: meteorology.sounding.Sounding instance
        """
        if file_path is not None:
            assert isinstance(file_path, str),\
                'file_path must be a string path to a file'
            assert os.path.isfile(file_path), \
                'file_path must point to a valid file'

            with open(file_path, 'r') as gsd_file:
                text = gsd_file.readlines()

        return self.from_text(text)

    def from_RUC_soundings(self,
        station=None, lat=None, lon=None,
        year=None, month=None, day=None, hour=None,
        model='Op40'):
        """Get and parse a sounding from the rucsoundings website

        Location can be specified either through station id or lat/lon. Station
        id takes preference. Time can either be specified or will use the
        current system time. Does not yet support getting multiple times

        Arguments:
            station: either specify a station or a lat/lon. Stations are WMO
                id or airport identifier
            lat: latitude, decimal degrees.
            lon: longitude, decimal degrees
            year: year, defaults to current system time if not specified
            month: month, defaults to current system time if not specified
            day: day, defaults to current system time if not specified
            hour: 24 hour clock, UTC, defaults to current system time if not
                specified
            model: model to pull from, defaults to Op40. Bak40 and GFS can
                also be specified

        Returns:
            sounding: meteorology.sounding.Sounding instance
        """
        now = datetime.datetime.now()
        if station is None:
            station = '{}%2C%20{}'.format(lat, lon)
        if year is None:
            year = now.year
        if month is None:
            month = now.month
        month_str = self._month_dict[month]
        if day is None:
            day = now.day
        if hour is None:
            hour = now.hour
        start = geodesy.conversions.datetime_to_unix(datetime.datetime(
            year, month, day, hour, 0, 0))
        end = geodesy.conversions.datetime_to_unix(datetime.datetime(
            year, month, day, hour+1, 0, 0)) + 1
        url = 'https://rucsoundings.noaa.gov/get_soundings.cgi?data_source=' +\
            model + \
            '&start_year=' + str(year) +\
            '&start_month_name=' + month_str +\
            '&start_mday=' + str(day) +\
            '&start_hour=' + str(hour) +\
            '&start_min=' + '0' +\
            '&n_hrs=1.0&fcst_len=shortest' +\
            '&airport=' + station +\
            '&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false' + \
            '&startSecs=' + str(start) +\
            '&endSecs=' + str(end)

        response = urllib2.urlopen(url)
        text = response.readlines()

        return self.from_text(text)

    def from_text(self, text):
        """Parse a sounding out of text

        Arguments:
            text: raw text containing the sounding to decode, a list
                with one entry for each line

        Returns:
            sounding: meteorology.sounding.Sounding instance
        """
        self._n_levels = 0
        self._P = []
        self._h = []
        self._T = []
        self._Ts = []
        self._u = []
        self._v = []

        #the first line is just info about the sounding
        self._info = text[0].replace('\n', '')

        # the second line contains date and time info
        data = re.search('(\w+)\s+'*5, text[1]).groups()
        self._source = data[0]
        self.date = datetime.date(
            int(data[4]),
            self._month_dict[data[3]],
            int(data[2]))
        self.hour = int(data[1])

        #bulk sounding info on the third line
        convective_data = re.search(
            'CAPE\s+([0-9]+)\s+' +
            'CIN\s+([-0-9]+)\s+' +
            'Helic\s+([0-9]+)\s+' +
            'PW\s+([0-9]+)', text[2]).groups()
        self.cape = int(convective_data[0])
        self.cin = int(convective_data[1])
        self.helicity = int(convective_data[2])
        self.pw = int(convective_data[3])

        #From here on out every line should have an identifier so use that
        for line in text[3:]:
            if len(line) < 7:
                continue
            line_id = self._get_line_type(line)
            if line_id in self._parser:
                self._parser[line_id](line)

        sounding_data = {
            'P': self._P,
            'z': self._h,
            'T': self._T,
            'dew_point': self._Ts,
            'u': self._u,
            'v': self._v,
            'year': int(data[4]),
            'month': self._month_dict[data[3]],
            'day': int(data[2]),
            'hour': int(data[1]),
            }
        sounding = meteorology.sounding.Sounding(sounding_data)
        return sounding

    def _parse_data_line(self, line):
        """Get the data from an all-integer line

        Many lines in the gsd file contain 7 integers, parse them apart

        Arguments:
            line: the line to parse, a string

        Returns:
            line_type: first integer identifies the line
            data: six integer list of data
        """
        str_data = re.search('\s+([\-0-9\.]+)'*7, line).groups()
        line_type = int(str_data[0])
        data = [float(d) for d in str_data[1:]]

        return (line_type, data)

    def _get_line_type(self, line):
        """Many lines have an integer idenfier, get that id

        Arguments:
            line: the line to parse

        Returns:
            line_type: integer idenfifying the line
        """
        str_data = re.search('\s+([0-9]+)', line).groups()[0]
        return int(str_data)

    def _parse_station_info(self, line):
        """Parse the line idenfying a station

        Arguments:
            line: line of text to parse

        Returns:
            no returns
        """
        line_id, data = self._parse_data_line(line)
        assert line_id == 1, 'station id must be line type 1'
        self.wban = data[0]
        self.wmo = data[1]
        self.lat = data[2]
        self.lon = data[3]
        self.alt = data[4]
        self.lla = numpy.array(data[2:5])

    def _parse_checks(self, line):
        """Parse the sounding checks line

        Arguments:
            line: line of text to parse

        Returns:
            no returns
        """
        line_id, data = self._parse_data_line(line)
        assert line_id == 2, 'sounding checks must be line type 1'
        self._n_levels = data[3]

    def _parse_station_id(self, line):
        """Parse the station id line

        Arguments:
            line: line of text to parse

        Returns:
            no returns
        """
        str_data = re.search('\s+([0-9a-zA-Z,-\.]+)'*4, line).groups()
        line_type = int(str_data[0])
        self.station_id = str_data[1]
        self.sonde_id = int(str_data[2])

        wind_sf = {
            'kt': 0.514444,
            'ms': 10.0
            }
        assert str_data[3] in wind_sf, 'wind units invalid'
        self._wind_scale = lambda x: x * wind_sf[str_data[3]]

        return

    def _parse_level(self, line):
        """Parse a line of sonde data

        Arguments:
            line: line of text to parse

        Returns:
            no returns
        """
        line_id, data = self._parse_data_line(line)

        self._n_levels -= 1.0

        if numpy.any(numpy.array(data[1:]) == 99999):
            return

        self._P.append(data[0] * 10.0)
        self._h.append(data[1])
        self._T.append(data[2] / 10.0)
        self._Ts.append(data[3] / 10.0)

        psi = numpy.deg2rad(data[4])
        U = self._wind_scale(data[5])
        self._u.append(numpy.sin(psi) * U)
        self._v.append(numpy.cos(psi) * U)

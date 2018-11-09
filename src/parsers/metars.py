import pdb

import os
import re
import copy
import datetime
import urllib2

import numpy

import geodesy.conversions

def get_metar(station, start_date, end_date=None):
    """Get metar data

    Arguments:
        station: string identifier of the station to get
        start_date: datetime instance giving the desired start of data
        end_date: datetime instance giving desired end of data. This is
            optional. If it is not specified then the metar closest to the
            date / hour given in start_date will be returned

    Returns:
        metar: numpy array of metar data
    """

    base_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'

    start_specification = 'year1={}&month1={}&day1={}'.format(
        start_date.year, start_date.month, start_date.day)
    if end_date is None:
        end_specification = '&year2={}&month2={}&day2={}'.format(
            start_date.year, start_date.month, start_date.day)
    else:
        end_specification = '&year2={}&month2={}&day2={}'.format(
            end_date.year, end_date.month, end_date.day)

    request_url = (
        base_url +
        'station={}&data=all&'.format(station.upper()) +
        start_specification +
        end_specification +
        '&tz=Etc%2FUTC&format=onlycomma' +
        '&latlon=yes&direct=no&report_type=1&report_type=2'
        )

    response = urllib2.urlopen(request_url)
    raw_metars = response.readlines()

    if raw_metars == '':
        return None

    metars = []
    for ia_metar_str in raw_metars:
        if 'station' in ia_metar_str:
            continue
        metars.append(Metar(ia_str=ia_metar_str))

    times = numpy.fromiter((m.gps_time for m in metars), dtype=float)
    if end_date is None:
        closest_idx = numpy.argmin(numpy.abs(
            times - geodesy.conversions.datetime_to_gps(start_date)))
        return metars[closest_idx]

    start_idx = None
    for idx, time in enumerate(times):
        if (
            time > geodesy.conversions.datetime_to_gps(start_date) and
            not start_idx):
            start_idx = idx
        if time < geodesy.conversions.datetime_to_gps(end_date):
            end_idx = idx

    return metars[start_idx:end_idx]

class Metar(object):
    """A class to hold a metar
    """
    def __init__(self, raw_str=None, ia_str=None):
        """Constructor

        Arguments:
            raw_str: raw metar string, like from ADDS
            ia_str: string of metar data as from Iowa state archive
                (https://mesonet.agron.iastate.edu/request/download.phtml?network=PA_ASOS)

        Returns:
            class instance
        """

        self._station = ''
        self._observation_time = numpy.nan
        self._T = numpy.nan
        self._dewpoint = numpy.nan
        self._RH = numpy.nan
        self._wind_direction = numpy.nan
        self._wind_speed = numpy.nan
        self._wind_gust = numpy.nan
        self._last_hour_precipitation = numpy.nan
        self._altimeter = numpy.nan
        self._mslp = numpy.nan
        self._visibility = numpy.nan
        self._sky_condition = [numpy.nan] * 4
        self._condition_altitudes = [numpy.nan] * 4
        self._codes = []
        self._location = []
        self._remarks = []

        self._sky_condition_dict = {
            'SKC': 0.0,
            'CLR': 0.0,
            'NSC': 0.0,
            'FEW': 0.25,
            'SCT': 0.5,
            'BKN': 0.875,
            'OVC': 1.0,
            'VV': -1.0}

        if raw_str:
            self.from_adds(raw_str)
            return

        if ia_str:
            self.from_ia(ia_str)

    def from_adds(self, year, month, metar_string):
        """Process metar from an ADDS format metar string

        Arguments:
            year: year of the metar ob
            month: month of the metar ob
            metar_string: string containing a metar

        Returns:
            no returns
        """
        fields = metar_string.split(' ')

        self._station = fields[0]

        observation_datetime = datetime.datetime(
            year,
            month,
            int(fields[1][:2]),
            int(fields[1][2:4]),
            int(fields[1][4:6]),
            0,
            0)
        self._observation_time = geodesy.conversions.datetime_to_gps(
            observation_datetime)

        self._wind_direction = numpy.deg2rad(float(fields[2][:3]))
        speed, gust = re.search('([0-9]+)G?([0-9]*)', fields[2][3:]).groups()
        if gust == '':
            gust = '0.0'
        self._wind_speed = float(speed) * 0.51444
        self._wind_gust = float(gust) * 0.51444

        visibility = fields[3].split('/')
        if len(visibility) > 1:
            self._visibility = float(visibility[0]) / float(visibility[1][0])
        else:
            visibility = visibility.split('SM')
            self._visibility = float(visibility[0])
        self._visibility *= 1609.34

        state = 'present_wx'
        last_state = 'present_wx'
        process_types = {
            'present_wx': self._process_present_wx,
            'sky_condition': self._process_sky_condition,
            'temp_dp': self._process_temp_dp,
            'altimeter': self._process_altimeter,
            'remarks': self._process_remarks,
            }

        for field in fields[4:]:
            if state in process_types:
                state = process_types[state](field)
            if last_state != state and state in process_types:
                state = process_types[state](field)
            last_state = copy.deepcopy(state)

    def _process_present_wx(self, field):
        """Process a present wx field

        Arguments:
            field: a field of present wx

        Returns:
            state: the state that the processor should now be in
        """
        # check to see if we got a sky condition field
        for key in self._sky_condition_dict.keys():
            if key in field:
                return 'sky_condition'

        self._codes.append(field)
        return 'present_wx'

    def _process_sky_condition(self, field):
        """Process a sky condition field

        Arguments:
            field: a field of sky condition

        Returns:
            state: the state the processor should now be in
        """
        # check to see if we got a temp/dp field
        if '/' in field:
            return 'temp_dp'

        condition, altitude = re.search('([A-Z]+)([0-9]*)', field).groups()
        self._sky_condition.append(self._sky_condition_dict[condition])
        if altitude == '':
            self._condition_altitudes.append(numpy.inf)
        else:
            self._condition_altitudes.append(float(altitude) * 1000.0 / 3.28)

    def _process_temp_dp(self, field):
        """Process a temp / dp field

        Arguments:
            field: field to process

        Returns:
            state: state the processor should now be in
        """
        if field[0] == 'A':
            return 'altimeter'

        temp, dp = field.split('/')

        self._T = float(temp.replace('M', '-')) + 273.15
        self._dewpoint = float(dp.replace('M', '-')) + 273.15

    def _process_altimeter(self, field):
        """Process an altimeter field

        Arguments:
            field: field to process

        Returns:
            state: the state the processor should now be in
        """
        if field[0] != 'A':
            return 'remarks'

        self._altimeter = float(field[1:]) / 100.0 * 33.8639

    def _process_remarks(self, field):
        """Process a remarks field

        Arguments:
            field: field to process

        Returns:
            state: the state the processor should now be in
        """
        self._remarks.append(field)
        return 'remarks'

    def from_ia(self, ia_str, latlon=True):
        """Process a metar from an Iowa state archive string

        See
        https://mesonet.agron.iastate.edu/request/download.phtml
        for more information about this format

        Arguments:
            ia_str: string from one line of a file requested from ia state
            latlon: optional, indicates if lat/lon included in the string.
                defaults true

        Returns:
            no returns
        """
        fields = ia_str.split(',')

        self._station = fields[0]
        observation_datetime = datetime.datetime.strptime(
            fields[1], '%Y-%m-%d %H:%M')
        self._observation_time = geodesy.conversions.datetime_to_gps(
            observation_datetime)

        offset = 0
        if latlon:
            self._location = numpy.deg2rad(numpy.array([
                float(fields[2]), float(fields[3])]))
            offset = 2

        self._T = (float(fields[2 + offset]) - 32.0) * 5.0 / 9.0 + 273.15
        self._dewpoint = (float(fields[3 + offset]) - 32.0) * 5.0 / 9.0 + 273.15
        self._RH = float(fields[4 + offset])
        self._wind_direction = numpy.deg2rad(float(fields[5 + offset]))
        self._wind_speed = float(fields[6 + offset]) * 0.51444
        self._last_hour_precipitation = float(fields[7 + offset]) * 2.54
        self._altimeter = float(fields[8 + offset]) * 33.8639
        if fields[9 + offset] == 'M':
            self._mslp = numpy.nan
        else:
            self._mslp = float(fields[9 + offset])
        self._visibility = float(fields[10 + offset]) * 1609.34
        self._wind_gust = float(fields[11 + offset].replace('M', '0.0'))
        #TODO: sky condition

    @property
    def T(self):
        """Getter for temperature

        Arguments:
            no arguments

        Returns:
            T: surface air temperature in K
        """
        return self._T

    @property
    def dewpoint(self):
        """Getter for dewpoint

        Arguments:
            no arguments

        Returns:
            dewpoint: return the dewpoint in K
        """
        return self._dewpoint

    @property
    def relative_humidity(self):
        """Getter for relative humidity

        Arguments:
            no arguments

        Returns:
            RH: relative humidity in %
        """
        return self._RH

    @property
    def wind_n(self):
        """Getter for north wind component (positive is air going north)

        Arguments:
            no arguments

        Returns:
            wind_n: wind component in north direction (m/s)
        """
        return -numpy.cos(self._wind_direction) * self._wind_speed

    @property
    def wind_e(self):
        """Getter for east wind component (postive is air going east)

        Arguments:
            no arguments

        Returns:
            wind_e: wind component in east direction (m/s)
        """
        return -numpy.sin(self._wind_direction) * self._wind_speed

    @property
    def altimeter(self):
        """Getter for altimeter setting

        Arguments:
            no arguments

        Returns:
            altimeter: return the altimeter setting in mb
        """
        return self._altimeter

    @property
    def gps_time(self):
        """Getter for gps time of this observation

        Arguments:
            no arguments

        Returns:
            gps_time: gps time at which this observation was made
        """
        return self._observation_time

    @property
    def unix_time(self):
        """Getter for unix time of this observation

        Arguments:
            no arguments

        Returns:
            unix_time: unix time at which this observation was made
        """
        unix_time = geodesy.conversions.gps_to_unix(self._observation_time)
        return unix_time

    @property
    def datetime(self):
        """Getter for datetime of this observation

        Arguments:
            no arguments

        Returns:
            datetime: datetime at which this observation was made
        """
        epoch = geodesy.conversions.gps_to_datetime(self._observation_time)
        return epoch

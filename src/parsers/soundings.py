import pdb

import os
import re
import copy
import datetime

import numpy

from meteorology.sounding import Sounding

def sounding_from_bufkit(buf_path=None, buf_string=None):
    """Create a sounding from bufkit file data

    Arguments:
        buf_path: path to file with ascii bufkit data, optional
        buf_string: string of data read in from an ascii bufkit file. This
            should be the raw output from e.g. file.read() (should not have
            lines split yet and should contain the line separators)

    Returns:
        sounding: environments.sounding.Sounding class instance
    """
    if buf_path:
        with open(buf_path, 'r') as buf_file:
            buf_string = buf_file.read()

    buf_parser = BufkitParser()
    sounding = buf_parser.parse_string(buf_string)
    return sounding

class BufkitParser(object):
    """A class to implement a parser for bufkit data
    """
    def __init__(self):
        """Constructor

        Arguments:
            no arguments

        Returns:
            no returns
        """
        # we'll use a finite state based read. define the state
        self._read_state = None
        self._sounding_fields = []
        self._station_fields = [
            'STID', 'STNM', 'TIME', 'SLAT', 'SLON', 'SELV', 'STIM']

        self._header_data = {}
        self._sounding_data = []
        self._data_idx = 0

        self._line_parsers = {
            None: self._file_header_parser,
            'header': self._header_parser,
            'surface_data': self._surface_data_parser,
            'sounding_data': self._sounding_data_parser,
            }
        self._state_transitions = {
            None: self._file_header_transitions,
            'header': self._header_transitions,
            'surface_data': self._surface_transitions,
            'sounding_data': self._sounding_transitions
            }

    def parse_string(self, string_data):
        """Parse string data into a sounding

        Arguments:
            string_data: a string containing bufkit file data. This should be
                the raw output of file.read() -- it should have the line
                terminators in with no splitting done yet

        Returns:
            sounding: environments.sounding.Sounding instance
        """
        data_lines = string_data.split('\r\n')

        for line in data_lines:
            self._read_state = self._state_transitions[self._read_state](line)

            self._line_parsers[self._read_state](line)

        soundings = []
        for sounding in self._sounding_data:
            M = numpy.array(sounding['SKNT'])
            psi = numpy.array(sounding['DRCT'])
            timestamp = sounding['header']['TIME']
            data = {
                'P': numpy.array(sounding['PRES']) * 100.0,
                'z': sounding['HGHT'],
                'T': numpy.array(sounding['TMPC']) + 273.15,
                'dew_point': numpy.array(sounding['DWPC']) + 273.15,
                'u': -numpy.sin(numpy.deg2rad(psi)) * M * 0.51444,
                'v': -numpy.cos(numpy.deg2rad(psi)) * M * 0.51444,
                'theta_e': sounding['THTE'],
                'year': timestamp.year,
                'month': timestamp.month,
                'day': timestamp.day,
                'hour': timestamp.hour,
                }
            soundings.append(Sounding(data))
        return soundings

    def _file_header_parser(self, line):
        """Parse a line from the file header

        Arguments:
            line: string data line

        Returns:
            no returns
        """
        line_id = line.split(' ')
        if line_id[0] == 'SNPARM':
            self._sounding_fields = line_id[2].split(';')
        if line_id[0] == 'STNPRM':
            self._station_fields.extend(line_id[2].split(';'))

    def _header_parser(self, line):
        """Parse a line from the header

        Arguments:
            line: string line data

        Returns:
            no returns
        """
        for field in self._station_fields:
            field_data = re.search(
                '{} = ([A-Za-z0-9\.\-/]+) '.format(field), line)
            if field_data:
                self._header_data[field] = self._process_header_field(
                    field, field_data.groups()[0])

    def _process_header_field(self, field_id, field_data):
        """Process a header field

        Arguments:
            field_id: name of the field to process
            field_data: data from the field

        Returns:
            data: processed data
        """
        if field_id == 'TIME':
            year = 2000 + int(field_data[:2])
            month = int(field_data[2:4])
            day = int(field_data[4:6])
            hour = int(field_data[7:9])
            minute = int(field_data[9:11])
            data = datetime.datetime(year, month, day, hour, minute, 0, 0)
        elif field_id == 'STID':
            data = field_data
        else:
            data = float(field_data)
        return data

    def _sounding_data_parser(self, line):
        """Parse a line from the sounding data

        Arguments:
            line: string containing a line of data from the file

        Returns:
            no returns
        """
        data = line.split(' ')
        if data[0] in self._sounding_fields or len(line) == 0:
            return
        for entry in data:
            latest_sounding = self._sounding_data[-1]
            latest_sounding[self._sounding_fields[self._data_idx]].append(
                float(entry))
            self._data_idx += 1
            self._data_idx %= len(self._sounding_fields)

    def _surface_data_parser(self, line):
        """Parse a line from the surface data

        Arguments:
            line: string containing a line of data from the file

        Returns:
            no returns
        """
        return

    def _file_header_transitions(self, line):
        """Transition from initial state

        Arguments:
            line: string data line

        Returns:
            new_state: new state to go to
        """
        line_id = line.split(' ')
        if len(line_id) < 1:
            return

        if line_id[0] == 'STID':
            return 'header'

    def _header_transitions(self, line):
        """Transition from header

        Arguments:
            line: string data line

        Returns:
            new_state: new state to go to
        """
        line_info = line.split(' ')
        if (
            line_info[-1] == self._sounding_fields[-1] and
            len(line_info[-1]) > 0):
            self._sounding_data.append({})
            for field in self._sounding_fields:
                self._sounding_data[-1][field] = []
            self._sounding_data[-1]['header'] = copy.deepcopy(self._header_data)
            self._header_data = {}
            return 'sounding_data'
        return 'header'

    def _sounding_transitions(self, line):
        """Transition from reading soundings

        Arguments:
            line: string data line

        Returns:
            new_state: new state to go to
        """
        line_info = line.split(' ')
        if line_info[0] == 'STN':
            return 'surface_data'
        if line_info[0] == 'STID':
            return 'header'
        return 'sounding_data'

    def _surface_transitions(self, line):
        return None

def from_RAOB(self, data, timestamp):
    """Create sounding from RAOB formatted data

    (ie the format that rucsoundings and UWyo give soundings in)

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
    data = {
        'P': data[:,0] * 100.0,
        'z': data[:,1],
        'T': data[:,2] + 273.15,
        'dew_point': data[:,3] + 273.15,
        'RH': data[:,4],
        'mixing_ratio': data[:,5],
        'u': -numpy.sin(numpy.deg2rad(data[:,6])) * data[:,7] * 0.51444,
        'v': -numpy.cos(numpy.deg2rad(data[:,6])) * data[:,7] * 0.51444,
        'theta': data[:,8],
        'theta_e': data[:,9],
        'theta_v': data[:,10],
        'year': timestamp['year'],
        'month': timestamp['month'],
        'day': timestamp['day'],
        'hour': timestamp['hour'],
        }

    return Sounding(data)

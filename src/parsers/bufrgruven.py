import pdb
import re
import numpy
import sys
import os

import environments.earth
import meteorology.sounding

class BufrGruvenParser(object):
    """parse and hold a GSD format sounding
    """
    def __init__(self):
        """Constructor

        Arguments:
            None

        Returns:
            class isntance
        """
        return

    def parse_ascii(self, file_path):
        """Parse a bufrgruven file

        Arguments:
            file_path: a path to a saved file

        Returns:
            soundings: list of meteorology.sounding.Sounding instances
        """
        assert os.path.isfile(file_path),\
            '{} is not a valid file'.format(file_path)

        with open(file_path, 'r') as sounding_file:
            data = sounding_file.readlines()

        sounding = []
        soundings = []
        for line in data:
            if re.search('STATION', line):
                soundings.append(self._from_ascii(sounding))
                sounding = []
            sounding.append(line)

        return soundings[1:]

    def _from_ascii(self, data):
        """Parse a sounding out from bufrgruven ascii
        """
        if len(data) == 0:
            return

        stamp = re.search('([0-9]{10})', data[1]).groups()[0]
        year = 2000 + int(stamp[:2])
        month = int(stamp[2:4])
        day = int(stamp[4:6])
        hour = int(stamp[6:8])

        layer_id = []
        temp = []
        dp = []
        kt = []
        direction = []
        pressure = []
        rh = []
        omega = []
        for data_field in data[5:-3]:
            entry = numpy.fromstring(data_field, sep=' ')
            layer_id.append(entry[0])
            temp.append(entry[1])
            dp.append(entry[2])
            kt.append(entry[3])
            direction.append(entry[4])
            pressure.append(entry[5])
            rh.append(entry[6])
            omega.append(entry[7])

        T = numpy.flipud(numpy.array(temp))
        dew_point = numpy.flipud(numpy.array(dp))
        wind_speed = numpy.flipud(numpy.array(kt))
        wind_speed *= 0.514444
        wind_direction = numpy.flipud(numpy.array(direction))
        rh = numpy.flipud(numpy.array(rh))
        pressure = numpy.flipud(numpy.array(pressure)) * 100.0

        u = numpy.sin(numpy.deg2rad(wind_direction)) * wind_speed
        v = numpy.cos(numpy.deg2rad(wind_direction)) * wind_speed

        # this is a little shady, but it generates a *rough* z
        # coordinate using the standard atmosphere since there's no z
        # contained in the bufrgruven ascii data
        atmosphere = environments.earth.Atmosphere()
        z = numpy.fromiter((
            atmosphere.hypsometric(
                T_bar=numpy.mean(273.15 + T[:idx+1]),
                P1=101325.0,
                P2=pressure[idx])
            for idx in range(len(T))), dtype=float)

        dict_data = {
                'P': pressure,
                'z': z,
                'T': T,
                'dew_point': dew_point,
                'u': u,
                'v': v,
                'RH': rh,
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                }

        sounding = meteorology.sounding.Sounding(dict_data)
        return sounding

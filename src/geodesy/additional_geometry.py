# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:12:40 2015

@author: nate
"""
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from geodesy.conversions import lla2ned
from geodesy.conversions import ned2lla

class GeoPolygon(object):
    """class holds both the lla and ned polygons that make up a shape on the
    map."""
    def __init__(self, coords=None, ref_pt=None):
        # set the reference point for this polygon
        if ref_pt is not None:
            self.ref_pt = ref_pt
        else:
            self.ref_pt = [0.0]*3
        # if we've been provided coordinates, set the polygons
        if coords is not None:
            # if we were given coordinates but no reference, assume lla
            if ref_pt is None:
                self.lla = coords
            else:
                # the provided coordinates must be ned as we have a ref point
                self.ned = coords
        else:
            # lla Polygon for this shape
            self._lla_shape = Polygon()
            # ned Polygon for this shape
            self._ned_shape = Polygon()
        # reference point (in lla) for the ned coordinate system for this shape
        # self._ref_pt = [0.0]*3

    def set_lla(self, poly):
        """Method sets the internal polygons when given a Polygon in lla
        coordinates

        Args:
            poly: shapely.geometry.Polygon object indicating the shape in lla
                coordinates (rad). Alternatively, poly may be an array of
                points defining the corners of the lla polygon
        """
        # check the input
        if type(poly) is not Polygon:
            # if we weren't given a polygon, turn the coordinates into one
            poly = Polygon(poly)
        # set the internal value for lla shape
        self._lla_shape = poly
        # get the vertex coordinates for the lla shape
        lla_coords_temp = np.array(self._lla_shape.exterior.xy).T
        # add a column of zeros for altitude (shapely is 2D..)
        lla_coords = np.zeros((lla_coords_temp.shape[0], 3))
        lla_coords[:,:-1] = lla_coords_temp
        # convert lla vertices to ned
        ned_coords = lla2ned(lla_coords, self._ref_pt)
        # make the ned shape out of these coordinates
        self._ned_shape = Polygon(ned_coords)

    def set_ned(self, input_poly, ref_pt=None):
        """
        Args:
            input_poly: shapely.geometry.Polygon object indicating the shape in
                ned coordinates (meters). Alternatively, poly may be an array
                of points defining the corners of the ned polygon
            ref_pt: 3 place list indicating reference point for ned polygon. If
                this value isn't provided, the object's current reference point
                is used
        """
        # check the input
        if type(input_poly) is not Polygon:
            # if we weren't given a polygon, turn the coordinates into one
            input_poly = Polygon(input_poly)
        # set ned shape to match input
        self._ned_shape = input_poly
        # set the reference point if provided
        self.ref_pt = ref_pt
        # get the vertex coordinates for the ned shape
        ned_coords_temp = np.array(self._ned_shape.exterior.xy).T
        # add a column of zeros for altitude (shapely is 2D..)
        ned_coords = np.zeros((ned_coords_temp.shape[0], 3))
        ned_coords[:,:-1] = ned_coords_temp
        # convert ned vertices to lla
        lla_coords = ned2lla(ned_coords, self._ref_pt)
        # make the ned shape out of these coordinates
        self._lla_shape = Polygon(lla_coords)

    def ned_pt(self, lla_pt):
        """Method converts an lla point to the ned frame defined by the current
        referenece point."""
        #ned_pt = lla2ned
        pass

    @property
    def ref_pt(self):
        """Returns the Polygon reference point as a 3 place list"""
        return self._ref_pt.flatten().tolist()

    @ref_pt.setter
    def ref_pt(self, input_pt):
        """Takes a three place list and sets the current reference point to
        equal this value."""
        if input_pt is not None:
            self._ref_pt = np.array([[0.0]*3])
            self._ref_pt[0] = input_pt

    @property
    def lla(self):
        """Returns the MapPolygon expresses in lla coordinates"""
        return self._lla_shape

    @lla.setter
    def lla(self, input_poly):
        """Method sets the internal polygons when given a Polygon in lla
        coordinates

        Args:
            input_poly: shapely.geometry.Polygon object indicating the shape in
                lla coordinates (rad). Alternatively, poly may be an array of
                points defining the corners of the lla polygon
        """
        # check the input
        if type(input_poly) is not Polygon:
        #if not isinstance(input_poly, Polygon):
            # if we weren't given a polygon, turn the coordinates into one
            if (type(input_poly) is np.ndarray) or (type(input_poly) is list):
                input_poly = Polygon(input_poly)
            else:
                return
        # set the internal value for lla shape
        self._lla_shape = input_poly
        # get the vertex coordinates for the lla shape
        lla_coords_temp = np.array(self._lla_shape.exterior.xy).T
        # add a column of zeros for altitude (shapely is 2D..)
        lla_coords = np.zeros((lla_coords_temp.shape[0], 3))
        lla_coords[:,:-1] = lla_coords_temp
        # convert lla vertices to ned
        ned_coords = lla2ned(lla_coords, self._ref_pt)
        # make the ned shape out of these coordinates
        ned_exterior = Polygon(ned_coords)

        # make a unified shape for the keep out zones
        keep_out_list = []
        for shape in input_poly.interiors:
            # convert keepout coords to ned
            shape = Polygon(shape)
            lla_coords_temp = np.array(shape.exterior.xy).T
            # add a column of zeros for altitude (shapely is 2D..)
            lla_coords = np.zeros((lla_coords_temp.shape[0], 3))
            lla_coords[:,:-1] = lla_coords_temp
            # convert lla vertices to ned
            ned_coords = lla2ned(lla_coords, self._ref_pt)
            # add this region to the list
            keep_out_list.append(Polygon(ned_coords) )
        keep_out = cascaded_union(keep_out_list)

        # now make a valid mission area polygon
        self._ned_shape = ned_exterior.difference(keep_out)


    @property
    def ned(self):
        """Returns the MapPolygon expresses in ned coordinates"""
        return self._ned_shape

    @ned.setter
    def ned(self, input_poly):
        """
        Args:
            input_poly: shapely.geometry.Polygon object indicating the shape in
                ned coordinates (meters). Alternatively, poly may be an array
                of points defining the corners of the ned polygon
        """
        # check the input
        if type(input_poly) is not Polygon:
        #if not isinstance(input_poly, Polygon):
            # if we weren't given a polygon, turn the coordinates into one
            if (type(input_poly) is np.ndarray) or (type(input_poly) is list):
                input_poly = Polygon(input_poly)
            else:
                return
        # set ned shape to match input
        self._ned_shape = input_poly
        # get the vertex coordinates for the ned shape
        ned_coords_temp = np.array(self._ned_shape.exterior.xy).T
        # add a column of zeros for altitude (shapely is 2D..)
        ned_coords = np.zeros((ned_coords_temp.shape[0], 3))
        ned_coords[:,:-1] = ned_coords_temp
        # convert ned vertices to lla
        lla_coords = ned2lla(ned_coords, self._ref_pt)
        # make the ned shape out of these coordinates
        self._lla_shape = Polygon(lla_coords)

    @property
    def area(self):
        """Returns the area of the GeoPolygon in meters^2 (because the other
        one doesn't make any damn sense)"""
        return self._ned_shape.area

class GeoPoint(object):
    """Class to hold a point in both lla and ned coordinates."""
    def __init__(self):
        self._lla_pt = Point()
        self._ned_pt = Point()
        self._ref_pt = np.array([[0.0]*3])

    def distance(self, other_pt, is_lla=True):
        """Method takes in another point and returns the unsigned distance
        between the two points. This method assumes that the compare point is
        given in lla (the really useful part of this class)."""
        return 0.0

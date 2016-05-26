import shapefile
import csv
import pandas as pd
import SVY21
from coordinatescrape import *

#### Note ####
"""
Extract subzone ID, central coordinate and area

make a FIND NEAREST SUBZONE function
"""

def get_midpoint(points):
	#get max and min of both lat and long
	lon = [point[0]for point in points]
	lat = [point[1]for point in points]
	lon = (max(lon) + min(lon)) / 2
	lat = (max(lat) + min(lat)) / 2
	return lon, lat

def get_area(points):
	# area of singapore is 719.1
	# 1 degree = 111.2km
	# 1 degree^2 = 12365.44km2

	# implementation of Green's Theorem to calculate area in polygon
	deg_to_km_constant = 12365.44
	total = 0.0
	N = len(points)
	for i in range(N):
		v1 = points[i]
		v2 = points[(i+1) % N]
		total += v1[0]*v2[1] - v1[1]*v2[0]
	return abs(total/2) * deg_to_km_constant

def open_shape(filename):
	# only opens the subzone shapefile with lon-lat data included
	sf = shapefile.Reader(filename)
	shapeRec = sf.shapeRecords()
	extract = []
	for i in range(len(shapeRec)):
		points = shapeRec[i].shape.points
		lon, lat = get_midpoint(points)
		area = get_area(points)
		subzone_ID = shapeRec[i].record[1]
		extract.append((subzone_ID, float(lon), float(lat), area)) 
	return extract

def open_shape2(filename):
	# used to extract coordinates by querying address on map api
	sf = shapefile.Reader(filename)
	shapeRec = sf.shapeRecords()
	extract = []
	for i in range(len(shapeRec)):
		locality = shapeRec[i].record[1]
		
		if '(' in locality:
			locality = locality[:locality.index('(')].strip()
		elif '/' in locality:
			locality = locality[:locality.index('/')].strip()

		lon, lat = scrap_coordinate(locality)
		case_number = shapeRec[i].record[2]
		idnum = shapeRec[i].record[1]
		extract.append((idnum, lon, lat, case_number))
	return extract

newlist = open_shape("shape/subzone.shp")
#print sorted(newlist)

#newlist = open_shape2("DailyData/230516/dengue-cluster/DENGUE_CLUSTER.shp")
#print newlist
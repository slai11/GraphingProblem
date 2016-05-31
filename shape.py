import math
import shapefile
import csv
import pandas as pd
from SVY21 import *
from coordinatescrape import *

#### Note ####
"""
Extract subzone ID, central coordinate and area
make a FIND NEAREST SUBZONE function
"""
def dist(A, B):
	x = A[0] - B[0]
	y = A[1] - B[1]
	return math.hypot(x,y)

def get_midpoint(points):
	#get max and min of both lat and long
	lon = [point[0]for point in points]
	lat = [point[1]for point in points]
	lon = (max(lon) + min(lon)) / 2
	lat = (max(lat) + min(lat)) / 2

	if lon > 200:
		cv = SVY21()
		lat, lon = cv.computeLatLon(lat, lon)

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


def open_scrap(filename):
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




#####################
# Data.gov shp files#
#####################
def load_breeding_habitats(habfilelist, subzonefile):
	sub = shapefile.Reader(subzonefile)
	subzone = sub.shapeRecords()
	extract = []
	for file in habfilelist: # files
		sf = shapefile.Reader(file)
		bh = sf.shapeRecords()

		for i in range(len(bh)): # number of bh
			parent = ""
			points = bh[i].shape.points
			lon, lat = get_midpoint(points)

			for j in range(len(subzone)): #number of subzones
				if is_in_area(lon, lat, subzone[j].shape.points):
					parent = subzone[j].record[1]
			
			extract.append((parent, 1))
	return collate_cases(extract)

def load_cluster_data(filename, subzonefile):
	sf = shapefile.Reader(filename)
	sub = shapefile.Reader(subzonefile)
	subzoneshapes = sub.shapeRecords()
	shapeRec = sf.shapeRecords()
	extract = []
	
	for i in range(len(shapeRec)):
		parent = ""
		points = shapeRec[i].shape.points
		cases = shapeRec[i].record[2] #tentative number
		
		lon, lat = get_midpoint(points)
		for j in range(len(subzoneshapes)):
			if is_in_area(lon, lat, subzoneshapes[j].shape.points):
				parent = subzoneshapes[j].record[1]
				break
				
		print parent + " " + str(cases)
		extract.append((parent, cases))
	
	return collate_cases(extract)
	
	'''
	target_lon, target_lat = get_midpoint(subzoneshapes[j].shape.points)
	distance = dist((lon, lat), (target_lon, target_lat))
	if not j:
		nearest = 1000
	(distance < nearest) and 
	nearest = distance
	'''

def collate_cases(caselist):
	newlist = {}
	for case, number in caselist:
		newlist[case] = 0
	for case, number in caselist:
		newlist[case] += number
	caselist = []
	caselist = [(k, v) for k, v in newlist.iteritems()]
	return caselist

def is_in_area(lon, lat, points):
	#check sum of angles == 360
	def dist(A, B):
		x = A[0] - B[0]
		y = A[1] - B[1]
		return math.hypot(x,y)

	def get_cos_theta(a, b, c):
		dist_c = dist(a, b)
		dist_a = dist(a, c)
		dist_b = dist(b, c)
		return (dist_c**2 - dist_a**2 - dist_b**2) / (-2.0 * dist_a * dist_b)

	N = len(points)
	C = (lon, lat)
	total_angle = 0
	for i in range(N):
		v1 = points[i]
		v2 = points[(i+1) % N]
		cos_theta = get_cos_theta(v1, v2, C)
		if cos_theta >= 1:
			angle = math.acos(1)
		else:
			angle = math.acos(cos_theta)
		total_angle += angle
	
	if total_angle > 6.28: # 6.28319:
		return True
	else:
		return False


if __name__ == '__main__':
	
	cluster = "DailyData/240516/dengue-clusters/DENGUE_CLUSTER.shp"

	bhlist = ['DailyData/230516/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
				'DailyData/230516/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
				 'DailyData/230516/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
				 'DailyData/230516/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
				 'DailyData/230516/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp',]
	
	shape = "shape/subzone.shp"

	#newlist = load_cluster_data(cluster, shape)
	newlist = load_breeding_habitats(bhlist, shape)
	print newlist
	#newdf = pd.DataFrame(newlist)
	#newdf.to_csv('2405cluster')

#newlist = open_shape("shape/subzone.shp")
#print sorted(newlist)

#newlist = open_shape2("DailyData/230516/dengue-cluster/DENGUE_CLUSTER.shp")
#print newlist
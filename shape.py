# _*_ coding: utf-8 _*_

import re
import math
import shapefile
import csv
import pandas as pd
from SVY21 import *
from coordinatescrape import *

"""
This file contains the classes related to opening shapefiles
- subzone shapefiles
- breeding habitat list
- dengue cases list
"""


class BasicExtractor(object):
	'''
	Superclass containing utility methods for finding:
	1. midpoint
	2. area of shape
	3. check if point is inside a larger area
	4. collates repeated list, summing its components
	'''
	def __init__(self):
		self.x = 10

	def dist(self, A, B):
		x = A[0] - B[0]
		y = A[1] - B[1]
		return math.hypot(x,y)

	def get_midpoint(self, points):
		#get max and min of both lat and long
		lon = [point[0]for point in points]
		lat = [point[1]for point in points]
		lon = (max(lon) + min(lon)) / 2
		lat = (max(lat) + min(lat)) / 2

		if lon > 200:
			cv = SVY21()
			lat, lon = cv.computeLatLon(lat, lon)

		return lon, lat

	def get_area(self, points):
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

	def is_in_area(self, lon, lat, points):
		#check sum of angles == 360	

		def get_cos_theta(a, b, c):
			dist_c = self.dist(a, b)
			dist_a = self.dist(a, c)
			dist_b = self.dist(b, c)
			return (dist_c**2 - dist_a**2 - dist_b**2) / (-2.0 * dist_a * dist_b)

		def is_within():
			pointslon = [point[0]for point in points]
			pointslat = [point[1]for point in points]
			if ( min(pointslon)< lon < max(pointslon)) and (min(pointslat)< lat <max(pointslat)):
				return True
			else:
				return False

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
		
		if total_angle > 6.28 and is_within():  # extra layer of check, midpoint must be within boundary# 6.28319:
			return True
		else:
			return False

	def collate(self, caselist):
		newlist = {}
		for case, number in caselist:
			newlist[case] = 0
		for case, number in caselist:
			newlist[case] += number
		caselist = []
		caselist = [(k, v) for k, v in newlist.iteritems()]
		return caselist

class SubzoneShapeExtractor(BasicExtractor):
	"""
	This class opens a subzone shapefile and finds the following attributes
	1. longitude
	2. latitude
	3. area
	4. region it belongs to
	5. planning area it belongs to

	main public method returns list of tuples with attributes
	"""
	def __init__(self, subzone, planning_area, region):
		# add some sort of boolean to know which resolution to open
		self.subzone = subzone
		self.list_ = self.open_shape(subzone, planning_area, region)

	def open_shape(self, filename, planning_area_file, region_file): #
		# only opens the subzone shapefile with lon-lat data included
		sf = shapefile.Reader(filename)
		shapeRec = sf.shapeRecords()
		extract = []
		for i in range(len(shapeRec)):
			points = shapeRec[i].shape.points
			lon, lat = super(SubzoneShapeExtractor,self).get_midpoint(points)
			area = super(SubzoneShapeExtractor, self).get_area(points)
			subzone_ID = shapeRec[i].record[1]
			planning_area = self.get_planning_area(lon, lat, planning_area_file)
			region = self.get_region(lon, lat, region_file)

			extract.append((subzone_ID, float(lon), float(lat), area, planning_area, region))
		
		return extract

	def get_planning_area(self, lon, lat, areafile):
		sf = shapefile.Reader(areafile)
		area = sf.shapeRecords()
		for i in range(len(area)):
			if super(SubzoneShapeExtractor, self).is_in_area(lon, lat, area[i].shape.points):
				return area[i].record[1]

	def get_region(self, lon, lat, regionfile):
		sf = shapefile.Reader(regionfile)
		region = sf.shapeRecords()
		for i in range(len(region)):
			if super(SubzoneShapeExtractor, self).is_in_area(lon, lat, region[i].shape.points):
				return region[i].record[1]



class BreedingHabitatExtractor(BasicExtractor):

	"""
	This class collates a list of breeding habitats in all 5 regions and collates
	list of subzone with number of breeding habitat and coordinates
	"""

	def __init__(self, breedinghabitat, shape):
		self.bh = breedinghabitat
		self.loclist_, self.fulllist_  = self.load_breeding_habitats(breedinghabitat, shape)
		

	def load_breeding_habitats(self, habfilelist, subzonefile):
		'''
		Opens list of breeding habitat file (5 zones) and does 2 tasks
			- get centroid coordinates of each BH
			- finds the subzone it belongs to and makes frequency list

		@returns list of subzone + # breeding habitats + lon lat bruh
		'''

		sub = shapefile.Reader(subzonefile)
		subzone = sub.shapeRecords()
		extract = []
		fulllist = []
		for file_ in habfilelist: # files
			sf = shapefile.Reader(file_)
			bh = sf.shapeRecords()

			for i in range(len(bh)): # number of bh
				parent = ""
				points = bh[i].shape.points
				lon, lat = super(BreedingHabitatExtractor, self).get_midpoint(points)

				for j in range(len(subzone)): #number of subzones
					if super(BreedingHabitatExtractor, self).is_in_area(lon, lat, subzone[j].shape.points):
						parent = subzone[j].record[1]
				
				extract.append((parent, 1)) # 1 is just a counter. DON'T CHANGE IT
				fulllist.append((float(lon), float(lat)))
		extract = super(BreedingHabitatExtractor, self).collate(extract)
		return extract, fulllist



class DengueCaseExtractor(BasicExtractor):
	'''
	This class takes in a list of case files + shapefile outputs 
	the subzone, with number of cases
	'''
	def __init__(self, cases, shape):
		self.cases = cases
		self.list_ = self.load_cases(cases, shape)

	def load_cases(self, cases, shape):
		casenumber = 0
		sub = shapefile.Reader(shape)
		subzone = sub.shapeRecords()
		extract = []
		for file_ in cases: # files
			sf = shapefile.Reader(file_)
			bh = sf.shapeRecords()

			for i in range(len(bh)): # number of bh
				parent = ""
				points = bh[i].shape.points
				lon, lat = super(DengueCaseExtractor, self).get_midpoint(points)

				for j in range(len(subzone)): #number of subzones
					if super(DengueCaseExtractor, self).is_in_area(lon, lat, subzone[j].shape.points):
						parent = subzone[j].record[1]
				
				case_number = self.__get_case_number(bh[i].record[1])
				extract.append((parent, case_number))
				
		extract = super(DengueCaseExtractor, self).collate(extract)
		return extract

	def __get_case_number(self, string):
		try:
			number = re.sub('[^0-9]' ,'', string)
		except:
			number = string
		return int(number)

	def load_cluster_data(self, filename, subzonefile):
		'''
		Opens ONE cluster shapefile and calculates list of cluster with their case number

		@returns list of subzone with # of cases
		'''

		sf = shapefile.Reader(filename)
		sub = shapefile.Reader(subzonefile)
		subzoneshapes = sub.shapeRecords()
		shapeRec = sf.shapeRecords()
		extract = []
		
		for i in range(len(shapeRec)):
			parent = ""
			points = shapeRec[i].shape.points
			cases = shapeRec[i].record[2] #tentative number
			
			lon, lat = super(DengueCaseExtractor, self).get_midpoint(points)
			for j in range(len(subzoneshapes)):
				if super(DengueCaseExtractor, self).is_in_area(lon, lat, subzone[j].shape.points):
					parent = subzoneshapes[j].record[1]
					break
					
			#print parent + " " + str(cases)
			extract.append((parent, cases))
		extract = super(DengueCaseExtractor, self).collate(extract)
		return extract #list (use numpy instead?)




if __name__ == '__main__':
	
	cluster = "DailyData/240516/dengue-clusters/DENGUE_CLUSTER.shp"

	bhlist = ['DailyData/230516/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
				'DailyData/230516/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
				 'DailyData/230516/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
				 'DailyData/230516/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
				 'DailyData/230516/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp',]
	
	caselist = ['DailyData/230516/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
				'DailyData/230516/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
				'DailyData/230516/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
				'DailyData/230516/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
				'DailyData/230516/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp']
	
	shape = "shape/subzone.shp"
	'''
	SE = SubzoneShapeExtractor(shape)
	print SE.list_

	BE = BreedingHabitatExtractor(bhlist, shape)
	print BE.caselist
	'''
	CE = DengueCaseExtractor(caselist, shape)
	print CE.caselist



	#newlist = load_cluster_data(cluster, shape)
	#newlist = load_breeding_habitats(bhlist, shape)
	
	#print newlist
	#newdf = pd.DataFrame(newlist)
	#newdf.to_csv('2405cluster')

#newlist = open_shape("shape/subzone.shp")
#print sorted(newlist)

#newlist = open_shape2("DailyData/230516/dengue-cluster/DENGUE_CLUSTER.shp")
#print newlist



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

def open_shape(filename): #
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


def open_scrap(filename): # google scraper
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
	return extractex




#####################
# Data.gov shp files#
#####################
def load_breeding_habitats(habfilelist, subzonefile):
	'''
	Opens list of breeding habitat file (5 zones) and does 2 tasks
		- get centroid coordinates of each BH
		- finds the subzone it belongs to and makes frequency list

	@returns list of subzone + # breeding habitats
	'''
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
	'''
	Opens ONE cluster shapefile and calculates list of cluster with their case number

	@returns list of subzone with # of cases
	'''

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
	
	return collate_cases(extract) #list (use numpy instead?)
	
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

def load_cluster_data(filename, subzonefile):
	'''
	Opens ONE cluster shapefile and calculates list of cluster with their case number

	@returns list of subzone with # of cases
	'''

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
	
	return collate_cases(extract) #list (use numpy instead?)
	
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

"""
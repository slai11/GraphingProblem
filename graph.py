# _*_ coding: utf-8 _*_

import csv
import numpy as np
import pandas as pd
import networkx as nx
#import itertools
import matplotlib.pyplot as plt
from coordinatescrape import *
from shape import *


# opening files
network_data = pd.read_csv("Data/network_20160511.csv")
wkend_network_data = pd.read_csv("Data/Original/20160514_network.csv")
subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")

class GraphGenerator():
	def __init__(self, nodes, network, bh=None):
		self.graph = nx.DiGraph()
		self.networkfile = network
		self.nodefile = nodes
		self.bhfile = bh

	def get_graphs(self):
		self.make_graph()

		# add breeding habitat data # IMPT
		'''
		breedinghabitat = pd.read_csv("Data/Processed/breedinghabitat.csv")
		lon = breedinghabitat['longitude']
		lat = breedinghabitat['latitude']
		'''

		lon = self.bhfile['Lon']
		lat = self.bhfile['Lat']

		BH = nx.Graph()
		original = nx.DiGraph(self.graph)
		for i in range(len(lon)):
			self.graph.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]),\
								weight=0.0, normweightmax=0.0, type=float(1),\
								area = float(0.5), population = float(1), popdensity = float(1),\
								hotspot = 0)
			BH.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]))

		return self.graph, original, BH

	def make_graph(self):
		self.networkfile = self.networkfile.drop(self.networkfile[self.networkfile.Source == self.networkfile.Target].index)
		
		self.networkfile['Weight'] = self.networkfile['Weight'].astype(float)

		self.networkfile['normweightbymax'] = (((self.networkfile['Weight']) - min(self.networkfile['Weight'])) /\
											(max(self.networkfile['Weight']) - min(self.networkfile['Weight'])))
		
		subset = self.networkfile[["Source", "Target", "normweightbymax"]] # no longer using weight
		edge_list = [tuple(x) for x in subset.values]
		self.graph.add_weighted_edges_from(edge_list)


		subset = self.nodefile["Subzone"]
		weight_set = self.nodefile["Cases"]
		#normweight_set = self.nodefile["normalize by sum"]
		normmaxweight_set = self.nodefile["Cases_Norm_Max"]
		lon_set = self.nodefile["Lon"]
		lat_set = self.nodefile["Lat"]
		area_set = self.nodefile["Area"]
		pop_set = self.nodefile["Population"]
		popdense_set = self.nodefile["Pop_density"]
		bh_count_set = self.nodefile["BH_count"]
		
		
		for i, subzone in enumerate(subset):
			self.graph.add_node(subzone, weight = float(weight_set[i]), \
						#normweightsum = float(normweight_set[i]),\
						normweightmax = float(normmaxweight_set[i]), \
						longitude = float(lon_set[i]),\
						latitude = float(lat_set[i]),\
						type = float(5 + 10*normmaxweight_set[i]),\
						area = float(area_set[i]),\
						population = float(pop_set[i]),\
						popdensity = float(popdense_set[i]),\
						bh_count = float(bh_count_set[i]),\
						hotspot = 1)

		# prune graph with zero degree centrality
		deg = nx.degree_centrality(self.graph)
		for node in self.graph.nodes():
			if deg[node] == 0:
				self.graph.remove_node(node)




















####################
# Graph Generation #
####################
DG = nx.DiGraph()

def clean_network_frame(df):
	#df = df.drop(df[<some boolean condition>].index)
	df = df.drop(df[df.Source == df.Target].index)
	return df

def make_graph(nodes, network):
	# input node and edge data into DiGraph
	cleaned_network = pd.DataFrame(clean_network_frame(network))
	make_edge(cleaned_network)
	#get_subzone_data(nodes)
	make_node(nodes)

def make_edge(network):
	# input direction into graph
	network['normweightbymax'] = (network['Weight'] - min(network['Weight'])) / \
								(max(network['Weight']) - min(network['Weight']))
	
	subset = network[["Source", "Target", "normweightbymax"]] # no longer using weight
	edge_list = [tuple(x) for x in subset.values]
	DG.add_weighted_edges_from(edge_list)

def make_node(nodes):
	# input node data into graph
	subset = nodes["subzone"]
	weight_set = nodes["cases"]
	normweight_set = nodes["normalize by sum"]
	normmaxweight_set = nodes["normalize by max"]
	lon_set = nodes["lon"]
	lat_set = nodes["lat"]
	area_set = nodes["area"]
	pop_set = nodes["population"]
	popdense_set = nodes["pop_density"]
	
	for i, subzone in enumerate(subset):
		DG.add_node(subzone, weight = float(weight_set[i]), \
					normweightsum = float(normweight_set[i]),\
					normweightmax = float(normmaxweight_set[i]), \
					longitude = float(lon_set[i]),\
					latitude = float(lat_set[i]),\
					type = float(5 + 10*normmaxweight_set[i]),\
					area = float(area_set[i]),\
					population = float(pop_set[i]),\
					popdensity = float(popdense_set[i]),\
					hotspot = 1)

def get_subzone_data(nodes):
	# extracting geospatial features from shapefile
	# run once only
	szlist = sorted(open_shape("shape/subzone.shp"))
	nodes['lon'] = [i[1] for i in szlist]
	nodes['lat'] = [i[2] for i in szlist]
	nodes['area'] = [i[3] for i in szlist]
	print nodes
	nodes.to_csv('subzonedatav2.csv')

def input_coor():
	# modify to take in list of tuple (town, lat, lon)
	coordinates = pd.read_csv("coordinates.csv")
	coordinates['original address'] = coordinates['original address'].apply(lambda x: x.replace(" singapore", ""))

	lat_dict = {}
	long_dict = {}
	lat = coordinates['latitude']
	longitude = coordinates['longitude']

	for i, address in enumerate(coordinates['original address']):
		lat_dict[address] = float(lat[i])
		long_dict[address] = float(longitude[i])
	return lat_dict, long_dict


def generate_subgraph(feature):
	def getX(x):
		if DG.node[x][feature]:
			return x
	opencases = [getX(x) for x in DG.nodes()]
	temp = DG.subgraph(opencases)
	return temp

########
# Util #
########

def output_dict(itemDict, filename):
	# export dictionary to a CSV
	with open(filename, 'wb') as filewriter:
		w = csv.writer(filewriter)

		for key in itemDict.iteritems():
			w.writerow(key)
		#w.writerow(itemDict.keys())
		#w.writerow(itemDict.values())


def transform_feature(df, df2, column_name):
	# number the nodes if needed
	transformer_dict = {}
	unique_value = set(df[column_name].tolist())
	# set dictionary key to be zone name and value to be index
	for i, value in enumerate(unique_value):
		transformer_dict[value] = i

	def label_map(y):
		return transformer_dict[y]
	
	# transform value in both data frames
	df[column_name] = df[column_name].apply( label_map )
	df2["Source"] = df2["Source"].apply( label_map )
	df2["Target"] = df2["Target"].apply( label_map )
	return df, df2

def prune_graph():
	'''
	remove nodes who are outliers or not contributing
	1. islands
	2. low traffic flow (zero) + low cases 
	'''
	deg = nx.degree_centrality(DG)
	for node in DG.nodes():
		if deg[node] == 0:
			DG.remove_node(node)
		

def get_graphs(feature = "weight"):
	make_graph(subzone_data, network_data)
	prune_graph()
	subDG = generate_subgraph(feature)

	breedinghab = pd.read_csv("Data/Processed/breedinghabitat.csv")
	lon = breedinghab['longitude']
	lat = breedinghab['latitude']

	BH = nx.Graph()
	originalDG = nx.DiGraph(DG)

	for i in range(len(lon)):
		DG.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]),\
					weight=0.0, normweightmax=0.0, normweightsum=0.0, type=float(1),\
					area = float(0.5), population = float(1), popdensity = float(1),\
					hotspot = 0)
		subDG.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]),\
					weight=0.0, normweightmax=0.0, normweightsum=0.0, type=float(1),\
					hotspot = 0)
		BH.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]))
	nx.write_gexf(DG, "fullcombinedgraphv3.gexf")
	return DG, subDG, BH, originalDG



if __name__ == '__main__':
	make_graph(subzone_data,wkend_network_data)


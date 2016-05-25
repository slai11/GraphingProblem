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
subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")

####################
# Graph Generation #
####################
DG = nx.DiGraph()

def make_graph(nodes, network):
	# input node and edge data into DiGraph
	make_edge(network)
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
					longitude = float(lon_set[i]), latitude = float(lat_set[i]),\
					type = float(5 + 10*normmaxweight_set[i]), area = float(area_set[i]),\
					population = float(pop_set[i]), popdensity = float(popdense_set[i]),\
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

def get_graphs(feature = "weight"):
	make_graph(subzone_data, network_data)
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



'''
########
# Main #
########

#transform_feature(subzone_data, network_data, "subzone")
make_graph(subzone_data, network_data)
subDG = generate_subgraph("weight") #only places with cases

print nx.info(subDG)

############################
# mapping breeding habitat #
############################
breedinghab = pd.read_csv("Data/Processed/breedinghabitat.csv")
lon = breedinghab['longitude']
lat = breedinghab['latitude']

for i in range(len(lon)):
	DG.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]),\
				weight=0.0, normweightmax=0.0, normweightsum=0.0, type=float(1),\
				area = float(0.5), population = float(1), popdensity = float(1))
	subDG.add_node(i, longitude = float(lon[i]), latitude = float(lat[i]),\
				weight=0.0, normweightmax=0.0, normweightsum=0.0, type=float(1))

print nx.info(subDG)

#d = nx.degree(subDG)
#nx.draw_networkx(subDG, nodelist = d.keys(), node_size = [v for v in d.values()],cmap = plt.cm.Blues, edge_cmap = plt.cm.Reds, width = 0.5)
#plt.show()
nx.write_gexf(DG, "fullcombinedgraphv2.gexf")
'''
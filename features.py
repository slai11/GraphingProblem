import math
import numpy as np 
import pandas as pd 
import networkx as nx
from graph import *

"""
normalize by max -> population density

add in own feature
1. infection pressure?
	- score of surrounding case weight and weighted flow rate
	- relative distance to breeding habitats and density of surrounding breeding habitats
2. degree?
3. eigenvector centrality
4. traffic flow (HITS)
5. pagerank
6. relative danger (10 nearest)
7. population
8. population density
"""

G, SG, BH, OG = get_graphs()

################################
# density & promixity analysis #
################################

def bh_proximity_density():

	def dist(i, node):
		x = BH.node[i]['longitude'] - OG.node[node]['longitude']
		y = BH.node[i]['latitude'] - OG.node[node]['latitude']
		# 1 degree = 111.2km
		return math.hypot(x,y) * 111.2

	for node in OG.nodes():
		bh_list = [dist(i, node) for i in BH.nodes()]
		bh_list = sorted(bh_list)
		
		# density portion 
		density = 0 # #breeding grounds within 2km radius
		for distance in bh_list:
			if distance < 2:
				density += 1
		
		# proximity component - average dist of 10 nearest bh
		distsum = 0.0
		for i, distance in enumerate(bh_list):
			while i < 10:
				distsum += distance
				i += 1
		distsum = distsum / 10

		index = density / distsum 
		#index = density * math.exp(distsum)
		OG.node[node]['BHPDI'] = index
		OG.node[node]['bh_density'] = density
		OG.node[node]['inverse_dist'] = math.exp(-distsum)


#####################
# Neighbor Analysis #
#####################

def bad_neighbour():
	'''
	pressure felt by receiving high volume of flow from active hotspots
	is the summation of product of weighted hotspot cases and bhpdi
	'''
	def get_sum_of_edge_in(edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			start = edge[0]
			to = edge[1]
			#edge_weight = OG[start][to]['weight']
			try:
				edge_weight = OG[to][start]['weight']
				sum_edge_weight += float(edge_weight)
			except:
				pass
		return sum_edge_weight

	def get_sum_of_edge_out(edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			sum_edge_weight += float(OG[edge[0]][edge[1]]['weight'])
		return sum_edge_weight

	def get_distance(edge):
		x = OG.node[edge[0]]['longitude'] - OG.node[edge[1]]['longitude']
		y = OG.node[edge[0]]['latitude'] - OG.node[edge[1]]['latitude']
		return math.hypot(x,y) * 111.2

	for node in OG.nodes():
		# generate breadth-first-search edge list 
		bfs_edge_list = list(nx.bfs_edges(OG, node))
		# get sum of edge
		sumedgein = get_sum_of_edge_in(bfs_edge_list)
		sumedgeout = get_sum_of_edge_out(bfs_edge_list)
		total_in_pressure =0.0
		total_out_pressure = 0.0
		for edge in bfs_edge_list:
			node_to_node_pressure = 0.0
			if edge[0] in node:
				# take data
				try:
					in_path_weight = OG[edge[1]][node]['weight'] / sumedgein
					cases = OG.node[edge[1]]['normweightmax']  # should we be using this?
					dist = get_distance(edge)
					popden = OG.node[edge[1]]['popdensity']
					node_to_node_pressure = 0.5 * (in_path_weight * cases * math.exp(-dist))
				except:
					pass
				out_path_weight = OG[node][edge[1]]['weight'] / sumedgeout
				bhpdi = OG.node[edge[1]]['BHPDI']
				out_pressure = 0.5 * (out_path_weight * math.exp(bhpdi))

			total_in_pressure += node_to_node_pressure
			total_out_pressure += out_pressure

		OG.node[node]['bad_neighbour_in'] = total_in_pressure
		OG.node[node]['bad_neighbour_out'] = total_out_pressure


########################
# Node & Link Analysis #
########################
def add_eigen_centrality():
	centrality = nx.eigenvector_centrality(OG)
	for node in OG.nodes():
		OG.node[node]['eigen_centrality'] = centrality[node]

def run_pagerank():
	nstart = {}
	for name in nx.nodes(OG):
		nstart[name] = OG.node[name]['normweightmax']
	
	pr = nx.pagerank(OG, weight = "normweightbymax")

	for node in OG.nodes():
		OG.node[node]['pagerank'] = pr[node]

def run_hits():
	# equivalent of traffic flow -> hub & authority
	nstart = {}
	for name in nx.nodes(OG):
		nstart[name] = OG.node[name]['normweightmax']
	#hits algo
	h,a = nx.hits(OG,max_iter = 30)
	for node in OG.nodes():
		OG.node[node]['hub'] = h[node]
		OG.node[node]['authority'] = a[node]

def generate_binary():
	for node in OG.nodes():
		if OG.node[node]['type'] == 1:
			OG.node[node]['passive_hotspot'] = 0
			OG.node[node]['active_hotspot'] = 0
		elif OG.node[node]['type'] == 5:
			OG.node[node]['passive_hotspot'] = 1
			OG.node[node]['active_hotspot'] = 0
		else:
			OG.node[node]['passive_hotspot'] = 0
			OG.node[node]['active_hotspot'] = 1

def build_feature():
	add_eigen_centrality() 
	run_hits() 
	run_pagerank()
	bh_proximity_density()
	generate_binary()
	bad_neighbour()

	x_list = []
	for area in OG.nodes():
		x_list.append((OG.node[area]['eigen_centrality'], \
					 OG.node[area]['pagerank'], OG.node[area]['hub'],\
					 OG.node[area]['authority'], OG.node[area]['population'],\
					 OG.node[area]['popdensity'], OG.node[area]['bad_neighbour_in'],\
					 OG.node[area]['bh_density'], OG.node[area]['inverse_dist'],\
					 OG.node[area]['bad_neighbour_out']))
	X = np.array(x_list)
	
	y_list = []
	for area in OG.nodes():
		y_list.append(OG.node[area]['active_hotspot'])
	y = np.array(y_list)

	return X, y

#build_feature()
#print OG.nodes(data=True)


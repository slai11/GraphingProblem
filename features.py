import math
import numpy as np 
import pandas as pd 
import networkx as nx
from graph import *



"""
normalize by max: -> lon, lat 
binary -> BH or hotspot or passive hotspot (just generate dummy)
	type = 1 BH, type = 5 passive, type > 5 active

normalize by max -> population density

add in own feature
1. infection pressure?*
	- score of surrounding case weight and weighted flow rate
	- relative distance to breeding habitats and density of surrounding breeding habitats
2. degree?
3. eigenvector centrality  X
4. traffic flow normalized by max -> scale with SKlearn (HITS)
5. pagerank
6. relative danger (10 nearest)*
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
		density = 0 # #breeding grounds within 2km radius
		
		# density portion 
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

		index = density / distsum  # function? 
		#index = density * math.exp(distsum)

		#print node + " " + str(index)
		OG.node[node]['BHPDI'] = index
		#OG.node[node][bh_density] = density
		#OG.node[node]['inverse_dist'] = 1 / distsum


def infection_pressure():
	# pressure felt by receiving high volume of flow from active hotspots
	# summation of weighted hotspot cases
	def get_sum_of_edge(edgelist):
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

	def get_distance(edge):
		x = OG.node[edge[0]]['longitude'] - OG.node[edge[1]]['longitude']
		y = OG.node[edge[0]]['latitude'] - OG.node[edge[1]]['latitude']
		return math.hypot(x,y) * 111.2

	infection_pressure_list = []

	for node in OG.nodes():
		# generate edge list
		bfs_edge_list = list(nx.bfs_edges(OG, node))
		# get sum of edge
		sumedge = get_sum_of_edge(bfs_edge_list)
		total_pressure =0.0
		for edge in bfs_edge_list:
			if edge[0] in node:
				# take data
				#path_weight = OG[node][edge[1]]['weight'] / sumedge
				try:
					path_weight = OG[edge[1]][node]['weight'] / sumedge
					cases = OG.node[node]['normweightmax']  # should we be using this?
					bhpdi = OG.node[node]['BHPDI']
					dist = get_distance(edge)

					node_to_node_pressure = path_weight * cases / dist  * bhpdi # tentative
				except:
					pass
			total_pressure += node_to_node_pressure

		OG.node[node]['infection_pressure'] = total_pressure




########################
# node & link analysis #
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
	infection_pressure()

	x_list = []
	for area in OG.nodes():
		x_list.append((OG.node[area]['BHPDI'], OG.node[area]['eigen_centrality'],\
					 OG.node[area]['pagerank'], OG.node[area]['hub'],\
					 OG.node[area]['authority'], OG.node[area]['population'],\
					 OG.node[area]['popdensity'], OG.node[area]['infection_pressure']))
	X = np.array(x_list)
	
	y_list = []
	for area in OG.nodes():
		y_list.append(OG.node[area]['active_hotspot'])
	y = np.array(y_list)

	return X, y

#build_feature()
#print OG.nodes(data=True)

'''
add_eigen_centrality(G)
generate_binary(G)
#print G.nodes(data=True)

print nx.info(G)
print nx.info(OG)
print nx.info(BH)

bh_proximity_density(BH, OG)
'''

import math
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
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

#G, SG, BH, OG = get_graphs()

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
					node_to_node_pressure = in_path_weight * cases * math.exp(-dist)
				except:
					pass
				out_path_weight = OG[node][edge[1]]['weight'] / sumedgeout
				bhpdi = OG.node[edge[1]]['BHPDI']
				out_pressure = out_path_weight * math.exp(bhpdi)

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

def add_betweenness_centrality():
	centrality = nx.betweenness_centrality(OG, weight = 'weight')
	for node in OG.nodes():
		OG.node[node]['betweenness_centrality'] = centrality[node]

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

def generate_tier():
	for node in OG.nodes():
		if OG.node[node]['weight'] == 0:
			OG.node[node]['active_hotspot'] = 0
		elif OG.node[node]['weight'] < 15 and OG.node[node]['weight'] != 0:
			OG.node[node]['active_hotspot'] = 1
		else:
			OG.node[node]['active_hotspot'] = 2

def build_feature():
	add_eigen_centrality()
	add_betweenness_centrality()
	run_hits() 
	run_pagerank()
	bh_proximity_density()
	bad_neighbour()

	generate_binary()
	#generate_tier()

	x_list = []
	for area in OG.nodes():
		x_list.append((OG.node[area]['eigen_centrality'],\
					OG.node[area]['betweenness_centrality'],\
					OG.node[area]['pagerank'],\
					OG.node[area]['hub'],\
					OG.node[area]['authority'],\
					OG.node[area]['population'],\
					OG.node[area]['popdensity'],\
					OG.node[area]['bad_neighbour_in'],\
					OG.node[area]['bh_density'],\
					OG.node[area]['inverse_dist'],\
					OG.node[area]['bad_neighbour_out']))
	
	X = np.array(x_list)
	
	y_list = []
	for area in OG.nodes():
		y_list.append(OG.node[area]['active_hotspot'])
	y = np.array(y_list)

	return X, y





class FeatureBuilder():

	def __init__(self, maingraph, originalgraph, breedinghabitat):
		self.G = maingraph
		self.OG = originalgraph
		self.BH = breedinghabitat
		self.dl = self.distance_to_all()
		self.build()


	def export_gexf(self, filename):
		nx.write_gexf(self.OG, filename)


	def build(self):
		self.centrality_analysis()
		self.link_analysis()
		self.bh_proximity_density()
		self.bad_neighbour()
		self.distance_to_all()
		#add in weekday-weekend H change & A change -> gives insights to the nature of the place (residential, work)
		

	def get_features(self):
		self.generate_binary()
		#self.generate_tier()

		x_list = []
		dist_list = []
		for area in self.OG.nodes():
			x_list.append((self.OG.node[area]['eigen_centrality'],\
						self.OG.node[area]['betweenness_centrality'],\
						self.OG.node[area]['pagerank'],\
						self.OG.node[area]['hub'],\
						self.OG.node[area]['authority'],\
						self.OG.node[area]['population'],\
						self.OG.node[area]['popdensity'],\
						self.OG.node[area]['bad_neighbour_in'],\
						self.OG.node[area]['bh_density'],\
						self.OG.node[area]['inverse_dist'],\
						self.OG.node[area]['bad_neighbour_out'],\
						self.OG.node[area]['hub_change'],\
						self.OG.node[area]['aut_change']))

			#dist_list = [tuple(x) for x in self.dist_graph.node]

		x1 = np.array(x_list)
		X = np.hstack((x1, self.dl))
		
		y_list = []
		for area in self.OG.nodes():
			y_list.append(self.OG.node[area]['active_hotspot'])
		y = np.array(y_list)

		return X, y

	def set_weekend_change(self, weekend):
		changelist = []
		for node in self.OG.nodes():
			self.OG.node[node]['hub_change']  = weekend.node[node]['hub'] - self.OG.node[node]['hub']
			self.OG.node[node]['aut_change']  = weekend.node[node]['authority'] - self.OG.node[node]['authority']
			

	def generate_binary(self):
		for node in self.OG.nodes():
			if self.OG.node[node]['type'] == 1:
				self.OG.node[node]['passive_hotspot'] = 0
				self.OG.node[node]['active_hotspot'] = 0
			elif self.OG.node[node]['type'] == 5:
				self.OG.node[node]['passive_hotspot'] = 1
				self.OG.node[node]['active_hotspot'] = 0
			else:
				self.OG.node[node]['passive_hotspot'] = 0
				self.OG.node[node]['active_hotspot'] = 1

	def generate_tier(self):
		for node in self.OG.nodes():
			if self.OG.node[node]['weight'] == 0:
				self.OG.node[node]['active_hotspot'] = 0
			elif self.OG.node[node]['weight'] < 15 and self.OG.node[node]['weight'] != 0:
				self.OG.node[node]['active_hotspot'] = 1
			else:
				self.OG.node[node]['active_hotspot'] = 2

	def centrality_analysis(self):
		eigen_centrality = nx.eigenvector_centrality(self.OG)
		btw_centrality = nx.betweenness_centrality(self.OG, weight = 'weight')
		for node in self.OG.nodes():
			self.OG.node[node]['eigen_centrality'] = eigen_centrality[node]
			self.OG.node[node]['betweenness_centrality'] = btw_centrality[node]

	def link_analysis(self):
		nstart = {}
		for name in nx.nodes(self.OG):
			nstart[name] = self.OG.node[name]['normweightmax']
		
		pr = nx.pagerank(self.OG, weight = "normweightbymax")
		h, a = nx.hits(self.OG, max_iter = 30)

		for node in self.OG.nodes():
			self.OG.node[node]['pagerank'] = pr[node]
			self.OG.node[node]['hub'] = h[node]
			self.OG.node[node]['authority'] = a[node]

	def bh_proximity_density(self):

		def dist(i, node):
			x = self.BH.node[i]['longitude'] - self.OG.node[node]['longitude']
			y = self.BH.node[i]['latitude'] - self.OG.node[node]['latitude']
			# 1 degree = 111.2km
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			bh_list = [dist(i, node) for i in self.BH.nodes()]
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
			self.OG.node[node]['BHPDI'] = index
			self.OG.node[node]['bh_density'] = density
			self.OG.node[node]['inverse_dist'] = math.exp(-distsum)

	def bad_neighbour(self):
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
				sum_edge_weight += float(self.OG[edge[0]][edge[1]]['weight'])
			return sum_edge_weight

		def get_distance(edge):
			x = self.OG.node[edge[0]]['longitude'] - self.OG.node[edge[1]]['longitude']
			y = self.OG.node[edge[0]]['latitude'] - self.OG.node[edge[1]]['latitude']
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			# generate breadth-first-search edge list 
			bfs_edge_list = list(nx.bfs_edges(self.OG, node))
			
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
						cases = self.OG.node[edge[1]]['normweightmax']  # should we be using this?
						dist = get_distance(edge)
						popden = self.OG.node[edge[1]]['popdensity']
						node_to_node_pressure = in_path_weight * cases * math.exp(-dist)
					except:
						pass
					out_path_weight = self.OG[node][edge[1]]['weight'] / sumedgeout
					bhpdi = self.OG.node[edge[1]]['BHPDI']
					out_pressure = out_path_weight * math.exp(bhpdi)

				total_in_pressure += node_to_node_pressure
				total_out_pressure += out_pressure

			self.OG.node[node]['bad_neighbour_in'] = total_in_pressure
			self.OG.node[node]['bad_neighbour_out'] = total_out_pressure
	
	def distance_to_all(self):
		def get_distance(source, target):
			x = self.OG.node[source]['longitude'] - self.OG.node[target]['longitude']
			y = self.OG.node[source]['latitude'] - self.OG.node[target]['latitude']
			return math.hypot(x,y) * 111.2
		dist_graph = nx.Graph()
		for source in self.OG.nodes():
			#self.dist_graph.add_node(source)
			for target in self.OG.nodes():
				dist = get_distance(source, target)
				dist_graph.add_edge(source,target,weight=dist)
		return nx.to_numpy_matrix(dist_graph, weight = 'weight')
		




if __name__ == '__main__':
	network_data = pd.read_csv("Data/network_20160511.csv")
	wkend_network_data = pd.read_csv("Data/Original/20160514_network.csv")
	subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")
	
	GG = GraphGenerator(network_data, subzone_data)
	GG2 = GraphGenerator(wkend_network_data, subzone_data)
	G, OG, BH = GG.get_graphs()
	WG, WOG, WBH = GG2.get_graphs()
	
	FB = FeatureBuilder(G, OG, BH)
	FB2 = FeatureBuilder(WG,WOG, WBH)
	FB.set_weekend_change(FB2.OG)
	X, y = FB.get_features()

	print len(X)
	'''
	X1, y1 = FB2.get_features()
	
	changelist = []
	for node in FB.OG.nodes():
		hubdiff = FB2.OG.node[node]['hub'] - FB.OG.node[node]['hub']
		autdiff = FB2.OG.node[node]['authority'] - FB.OG.node[node]['authority']
		changelist.append((node,hubdiff, autdiff))
	
	df = pd.DataFrame(changelist)
	

	edgechangelist = []
	for n, nbrs in OG.adjacency_iter():
		for nbr, eattr in nbrs.items():
			change = 0.0
			try:
				change = WOG[n][nbr]['weight'] - OG[n][nbr]['weight']
			except:
				pass
			edgechangelist.append((n, nbr, change))

	df2 = pd.DataFrame(edgechangelist)
	print df2.sort_values(2)
	'''


	#print df
	
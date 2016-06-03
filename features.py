# _*_ coding: utf-8 _*_

import math
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.base_estimator import BaseEstimator
from graph import *

"""

"""



class BadNeighbours(BaseEstimator):

	def __init__(self, originalgraph, breedinghabitat, second_degree=True):
		self.second_degree = second_degree
		self.OG = originalgraph

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		self.__bh_proximity_density()
		self.__bad_neighbour()
		# returns a numpy array 
		featurelist = []

		if self.second_degree:
			self.__second_order_bad_neighbour()
			for node in self.OG.nodes():
				featurelist.append((self.OG.node[area]['bad_neighbour_in'],\
									self.OG.node[area]['bad_neighbour_out'],\
									self.OG.node[area]['2nd_bad_neighbour_in'],\
									self.OG.node[area]['2nd_bad_neighbour_out']))
		else:
			for node in self.OG.nodes():
				featurelist.append((self.OG.node[area]['bad_neighbour_in'],\
									self.OG.node[area]['bad_neighbour_out']))
		
		return np.array(featurelist)


	def __bh_proximity_density(self):
		def dist(i, node):
			x = self.BH.node[i]['longitude'] - self.OG.node[node]['longitude']
			y = self.BH.node[i]['latitude'] - self.OG.node[node]['latitude']
			# 1 degree = 111.2km
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			bh_list = [dist(i, node) for i in self.BH.nodes()]
			bh_list = sorted(bh_list)
			
			# density portion 
			density = 0 # breeding grounds within 2km radius
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
			self.OG.node[node]['BHPDI'] = index

	def __get_sum_of_edge_in(self, edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			start = edge[0]
			to = edge[1]
			#edge_weight = OG[start][to]['weight']
			try:
				edge_weight = OG[to][start]['weight']
				sum_edge_weight += float(edge_weight)
			except:
				sum_edge_weight += 0
		return sum_edge_weight

	def __get_sum_of_edge_out(self, edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			sum_edge_weight += float(self.OG[edge[0]][edge[1]]['weight'])
		return sum_edge_weight

	def __bad_neighbour(self):
		'''
		pressure felt by receiving high volume of flow from active hotspots
		is the summation of product of weighted hotspot cases and bhpdi
		'''
		def get_distance(edge):
			x = self.OG.node[edge[0]]['longitude'] - self.OG.node[edge[1]]['longitude']
			y = self.OG.node[edge[0]]['latitude'] - self.OG.node[edge[1]]['latitude']
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			# generate breadth-first-search edge list 
			bfs_edge_list = list(nx.bfs_edges(self.OG, node))
			
			# get sum of edge
			sumedgein = self.__get_sum_of_edge_in(bfs_edge_list)
			sumedgeout = self.__get_sum_of_edge_out(bfs_edge_list)
			total_in_pressure =0.0
			total_out_pressure = 0.0
			
			for edge in bfs_edge_list:
				node_to_node_pressure = 0.0
				if edge[0] in node:
					# take data
					try:
						in_path_weight = self.OG[edge[1]][node]['weight'] / sumedgein
						cases = self.OG.node[edge[1]]['normweightmax']  # should we be using this?
						dist = get_distance(edge)
						popden = self.OG.node[edge[1]]['popdensity']
					except:
						pass
						print "passed"
					node_to_node_pressure = in_path_weight * cases * math.exp(-dist)
					out_path_weight = self.OG[node][edge[1]]['weight'] / sumedgeout
					bhpdi = self.OG.node[edge[1]]['BHPDI']
					out_pressure = out_path_weight * math.exp(bhpdi)

				total_in_pressure += node_to_node_pressure
				total_out_pressure += out_pressure

			self.OG.node[node]['bad_neighbour_in'] = total_in_pressure
			self.OG.node[node]['bad_neighbour_out'] = total_out_pressure
	
	def __second_order_bad_neighbour(self):
		for node in self.OG.nodes():
			#calculate the weighted sum of "bad neighbour in" score
			bfs_edge_list = list(nx.bfs_edges(self.OG, node))
			sumedgein = self.__get_sum_of_edge_in(bfs_edge_list)
			sumedgeout = self.__get_sum_of_edge_out(bfs_edge_list)
			total_in_pressure = 0.0
			total_out_pressure = 0.0

			for edge in bfs_edge_list:
				# weight * bni-score 
				# ignore if from node
				try:
					bni_score = self.OG.node[edge[1]]['bad_neighbour_in']
					bno_score = self.OG.node[edge[1]]['bad_neighbour_out']
					in_weight = self.OG[edge[1]][node]['weight'] / sumedgein
					out_weight = self.OG[node][edge[1]]['weight'] / sumedgeout
					total_in_pressure += in_weight * bni_score
					total_out_pressure += out_weight * bno_score
				except:
					pass

			self.OG.node[node]['2nd_bad_neighbour_in'] = total_in_pressure
			self.OG.node[node]['2nd_bad_neighbour_out'] = total_out_pressure


class HitsChange(BaseEstimator):
	'''
	Only includes change in hub and authority score (represents flow rate)
	'''

	def __init__(self, normal ,weekend):
		self.OG = normal
		self.WOG = weekend

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		self.__link_analysis()
		self.__set_weekend_change()

		featurelist = []

		for node in self.OG.nodes():
			featurelist.append((self.OG.node[area]['hub_change'],\
								self.OG.node[area]['aut_change']))

		return np.array(featurelist)

	def __link_analysis(self):
		nstart = {}
		for name in nx.nodes(self.OG):
			nstart[name] = self.OG.node[name]['normweightmax']
		
		h, a = nx.hits(self.OG, max_iter = 30)

		for node in self.OG.nodes():
			self.OG.node[node]['hub'] = h[node]
			self.OG.node[node]['authority'] = a[node]

	def __set_weekend_change(self, weekend):
		changelist = []
		for node in self.OG.nodes():
			self.OG.node[node]['hub_change']  = self.WOG.node[node]['hub'] - self.OG.node[node]['hub']
			self.OG.node[node]['aut_change']  = self.WOG.node[node]['authority'] - self.OG.node[node]['authority']


class GeospatialEffect(BaseEstimator):

	def __init__(self, graph):
		self.OG = graph

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		def get_distance(source, target):
			x = self.OG.node[source]['longitude'] - self.OG.node[target]['longitude']
			y = self.OG.node[source]['latitude'] - self.OG.node[target]['latitude']
			return math.hypot(x,y) * 111.2
		
		dist_graph = nx.Graph()
		for source in self.OG.nodes():
			#self.dist_graph.add_node(source)
			dist = 0.0
			for target in self.OG.nodes():
				if source is not target:
					dist = get_distance(source, target)
					dist_graph.add_edge(source,target,weight= 1.0/dist)
				else:
					dist_graph.add_edge(source,target, weight = 0.0)
		return nx.to_numpy_matrix(dist_graph, weight = 'weight')


#G, SG, BH, OG = get_graphs()
class BasicFeatureBuilder():
	def __init__(self, maingraph, originalgraph, breedinghabitat):
		self.G = maingraph
		self.OG = originalgraph
		self.BH = breedinghabitat
		#self.dl = self.__distance_to_all()
		self.__build()


	def export_gexf(self, filename):
		nx.write_gexf(self.OG, filename)



	def __build(self):
		self.__centrality_analysis()
		self.__link_analysis()
		self.__bh_proximity_density()
		self.__bad_neighbour()
		self.__second_order_bad_neighbour()
		#add in weekday-weekend H change & A change -> gives insights to the nature of the place (residential, work)
		

	def get_features(self):
		self.__generate_binary()
		#self.__generate_tier()

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
						self.OG.node[area]['bh_density'],\
						self.OG.node[area]['inverse_dist'],\
						# can remove from here on aft OOP-ing
						self.OG.node[area]['bad_neighbour_in'],\
						self.OG.node[area]['bad_neighbour_out'],\
						self.OG.node[area]['2nd_bad_neighbour_in'],\
						self.OG.node[area]['2nd_bad_neighbour_out'],\
						self.OG.node[area]['hub_change'],\
						self.OG.node[area]['aut_change']))

		X = np.array(x_list)
		#X = self.dl
		#X = np.hstack((x1, self.dl))
		
		y_list = []
		for area in self.OG.nodes():
			y_list.append(self.OG.node[area]['active_hotspot'])
		y = np.array(y_list)

		return X, y

	def get_features_wo_change(self):
		self.__generate_binary()
		#self.__generate_tier()

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
						self.OG.node[area]['bh_density'],\
						self.OG.node[area]['inverse_dist'],\
						#can remove from here on aft OOP-ing 
						self.OG.node[area]['bad_neighbour_in'],\
						self.OG.node[area]['bad_neighbour_out'],\
						self.OG.node[area]['2nd_bad_neighbour_in'],\
						self.OG.node[area]['2nd_bad_neighbour_out']\
						))

		X = np.array(x_list)
		
		y_list = []
		for area in self.OG.nodes():
			y_list.append(self.OG.node[area]['active_hotspot'])
		y = np.array(y_list)

		return X, y

	def set_weekend_change(self, weekend):
		changelist = []
		for node in self.OG.nodes():
			self.OG.node[node]['hub_change']  = (weekend.node[node]['hub'] - self.OG.node[node]['hub']) / self.OG.node[node]['hub']
			self.OG.node[node]['aut_change']  = (weekend.node[node]['authority'] - self.OG.node[node]['authority']) / self.OG.node[node]['authority']
			
	def __generate_binary(self):
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

	def __generate_tier(self):
		for node in self.OG.nodes():
			if self.OG.node[node]['weight'] == 0:
				self.OG.node[node]['active_hotspot'] = 0
			elif self.OG.node[node]['weight'] < 15 and self.OG.node[node]['weight'] != 0:
				self.OG.node[node]['active_hotspot'] = 1
			else:
				self.OG.node[node]['active_hotspot'] = 2

	def __centrality_analysis(self):
		eigen_centrality = nx.eigenvector_centrality(self.OG, weight = 'weighted')
		btw_centrality = nx.betweenness_centrality(self.OG, weight = 'weight')
		for node in self.OG.nodes():
			self.OG.node[node]['eigen_centrality'] = eigen_centrality[node]
			self.OG.node[node]['betweenness_centrality'] = btw_centrality[node]

	def __link_analysis(self):
		nstart = {}
		for name in nx.nodes(self.OG):
			nstart[name] = self.OG.node[name]['normweightmax']
		
		pr = nx.pagerank(self.OG, weight = "normweightbymax")
		h, a = nx.hits(self.OG, max_iter = 30)

		for node in self.OG.nodes():
			self.OG.node[node]['pagerank'] = pr[node]
			self.OG.node[node]['hub'] = h[node]
			self.OG.node[node]['authority'] = a[node]

	def __bh_proximity_density(self):

		def dist(i, node):
			x = self.BH.node[i]['longitude'] - self.OG.node[node]['longitude']
			y = self.BH.node[i]['latitude'] - self.OG.node[node]['latitude']
			# 1 degree = 111.2km
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			bh_list = [dist(i, node) for i in self.BH.nodes()]
			bh_list = sorted(bh_list)
			
			# density portion 
			density = 0 # breeding grounds within 2km radius
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





	def __get_sum_of_edge_in(self, edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			start = edge[0]
			to = edge[1]
			# edge_weight = OG[start][to]['weight']
			#edge_weight = self.OG[to][start]['weight']
			#sum_edge_weight += float(edge_weight)
			try:
				edge_weight = self.OG[to][start]['weight']
				sum_edge_weight += float(edge_weight)
			except:
				sum_edge_weight += 0
		return sum_edge_weight

	def __get_sum_of_edge_out(self, edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			sum_edge_weight += float(self.OG[edge[0]][edge[1]]['weight'])
		return sum_edge_weight

	def __bad_neighbour(self):
		'''
		pressure felt by receiving high volume of flow from active hotspots
		is the summation of product of weighted hotspot cases and bhpdi
		'''

		def get_distance(edge):
			x = self.OG.node[edge[0]]['longitude'] - self.OG.node[edge[1]]['longitude']
			y = self.OG.node[edge[0]]['latitude'] - self.OG.node[edge[1]]['latitude']
			return math.hypot(x,y) * 111.2

		for node in self.OG.nodes():
			# generate breadth-first-search edge list 
			bfs_edge_list = list(nx.bfs_edges(self.OG, node))
			
			# get sum of edge
			sumedgein = self.__get_sum_of_edge_in(bfs_edge_list)
			sumedgeout = self.__get_sum_of_edge_out(bfs_edge_list)

			#store sumedge data
			self.OG.node[node]['sum_edge_in'] = sumedgein
			self.OG.node[node]['sum_edge_out'] = sumedgeout

			total_in_pressure = 0.0
			total_out_pressure = 0.0
			
			for edge in bfs_edge_list:
				node_to_node_pressure = 0.0
				if edge[0] in node:
					# take data
					try:
						in_path_weight = self.OG[edge[1]][node]['weight'] / sumedgein
						cases = self.OG.node[edge[1]]['normweightmax']  # should we be using this?
						dist = get_distance(edge)
						popden = self.OG.node[edge[1]]['popdensity']
						node_to_node_pressure = in_path_weight * cases * math.exp(-dist)
					except:
						node_to_node_pressure = 0
						
					out_path_weight = self.OG[node][edge[1]]['weight'] / sumedgeout
					bhpdi = self.OG.node[edge[1]]['BHPDI']
					out_pressure = out_path_weight * math.exp(bhpdi)

				total_in_pressure += node_to_node_pressure
				total_out_pressure += out_pressure

			self.OG.node[node]['bad_neighbour_in'] = total_in_pressure
			self.OG.node[node]['bad_neighbour_out'] = total_out_pressure
	
	def __second_order_bad_neighbour(self):  
		
		for node in self.OG.nodes():
			
			#calculate the weighted sum of "bad neighbour in" score
			bfs_edge_list = list(nx.bfs_edges(self.OG, node))
			sumedgein = self.__get_sum_of_edge_in(bfs_edge_list)
			sumedgeout = self.__get_sum_of_edge_out(bfs_edge_list)
			total_in_pressure = 0.0
			total_out_pressure = 0.0

			for edge in bfs_edge_list:
				# weight * bni-score 
				# ignore if from node
				try:
					# put a test here??
					bni_score = self.OG.node[edge[1]]['bad_neighbour_in']
					in_weight = self.OG[edge[1]][node]['weight'] / sumedgein
					# get correct sumedge (bfs tree again)
					bni_score = self.__clean_bni(bni_score, node, edge[1])
					total_in_pressure += in_weight * bni_score
					

					bno_score = self.OG.node[edge[1]]['bad_neighbour_out']
					out_weight = self.OG[node][edge[1]]['weight'] / sumedgeout
					total_out_pressure += out_weight * bno_score
				except:
					total_out_pressure += 0
				

			self.OG.node[node]['2nd_bad_neighbour_in'] = total_in_pressure
			self.OG.node[node]['2nd_bad_neighbour_out'] = total_out_pressure

	def __clean_bni(self,score, source, target):
		def get_distance(source, target):
			x = self.OG.node[source]['longitude'] - self.OG.node[target]['longitude']
			y = self.OG.node[source]['latitude'] - self.OG.node[target]['latitude']
			return math.hypot(x,y) * 111.2
		
		edge = self.OG[source][target]['weight']
		# Original formula node_to_node_pressure = in_path_weight * cases * math.exp(-dist)
		dist = get_distance(source, target)
		target_sum_edge_in = self.OG.node[target]['sum_edge_in']
		source_contribution =  (edge/target_sum_edge_in) * self.OG.node[source]['normweightmax'] * math.exp(-dist)
		correct = score - source_contribution
		return correct




	def __distance_to_all(self):
		def get_distance(source, target):
			x = self.OG.node[source]['longitude'] - self.OG.node[target]['longitude']
			y = self.OG.node[source]['latitude'] - self.OG.node[target]['latitude']
			return math.hypot(x,y) * 111.2
		
		dist_graph = nx.Graph()
		for source in self.OG.nodes():
			#self.dist_graph.add_node(source)
			dist = 0.0
			for target in self.OG.nodes():
				if source is not target:
					dist = get_distance(source, target)
					dist_graph.add_edge(source,target,weight= 1.0/dist)
				else:
					dist_graph.add_edge(source,target, weight = 0.0)
		return nx.to_numpy_matrix(dist_graph, weight = 'weight')
		




if __name__ == '__main__':
	network_data = pd.read_csv("Data/network_20160511.csv")
	wkend_network_data = pd.read_csv("Data/Original/20160514_network.csv")
	subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")

	GG = GraphGenerator(network_data, subzone_data)
	GG2 = GraphGenerator(wkend_network_data, subzone_data)
	G, OG, BH = GG.get_graphs()
	WG, WOG, WBH = GG2.get_graphs()
<<<<<<< HEAD

	FB = FeatureBuilder(G, OG, BH)
	FB2 = FeatureBuilder(WG,WOG, WBH)
=======
	
	FB = BasicFeatureBuilder(G, OG, BH)
	FB2 = BasicFeatureBuilder(WG,WOG, WBH)
>>>>>>> 10cb3ee9f02f38e16a682058acb91280c2849c03
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
	
# _*_ coding: utf-8 _*_

import math
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion
from graph import *


class BadNeighbours(BaseEstimator):
	"""
	Bad Neighbour calculates the 2 variations of effects which a neighbouring node can have on a node. 
	IN : weighted sum of the active cases and exp(-dist) for neighbouring nodes (weighted by inflow)
	OUT: weighted sum of the breedinghabitat-proximity divded by density for neighbouring nodes (weighted by outflow)

	Returns a np.array of bni, bno, bn2i, bn2o variables
	"""

	def __init__(self, originalgraph, breedinghabitat, tobuild=True ,second_degree=True):
		self.second_degree = second_degree
		self.OG = originalgraph
		self.BH = breedinghabitat

		#self.__bh_proximity_density()
		if tobuild:
			self.__bad_neighbour()
			if self.second_degree:
				self.__second_order_bad_neighbour()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		#self.__bh_proximity_density()
		#fromself.__bad_neighbour()
		# returns a numpy array 
		featurelist = []

		if self.second_degree:
			for node in self.OG.nodes():
				featurelist.append((self.OG.node[node]['bad_neighbour_in'],\
									self.OG.node[node]['bad_neighbour_out'],\
									self.OG.node[node]['bad_neighbour_in2'],\
									self.OG.node[node]['bad_neighbour_out2']))
		else:
			for node in self.OG.nodes():
				featurelist.append((self.OG.node[node]['bad_neighbour_in'],\
									self.OG.node[node]['bad_neighbour_out']))
		
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
			#index = density * math.exp(distsum)
			self.OG.node[node]['BHPDI'] = index
			self.OG.node[node]['bh_density'] = density
			self.OG.node[node]['inverse_dist'] = math.exp(-distsum)


	def __get_sum_of_edge_in(self, edgelist):
		sum_edge_weight = 0.0
		for edge in edgelist:
			start = edge[0]
			to = edge[1]
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
				

			self.OG.node[node]['bad_neighbour_in2'] = total_in_pressure
			self.OG.node[node]['bad_neighbour_out2'] = total_out_pressure

	def __clean_bni(self,score, source, target):
		'''
		This method removes the contribution of the source node to the target node's BNI 
		when calculating BN2I.
		'''
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


class HitsChange(BaseEstimator): #might just throw this aside???
	'''
	Only includes change in hub and authority score (represents flow rate)
	'''

	def __init__(self, normal ,weekend):
		self.OG = normal
		self.WOG = weekend
		self.__link_analysis()
		self.__set_weekend_change()

	def fit(self, X=None, y=None):
		return self

	def transform(self, X=None):
		featurelist = []

		for node in self.OG.nodes():
			featurelist.append((self.OG.node[node]['hub_change'],\
								self.OG.node[node]['aut_change']))

		return np.array(featurelist)

	def __link_analysis(self): # recalculates hub and authority rate

		# insert check for existing hub? reduce computational time
		nstart = {}
		for name in nx.nodes(self.OG):
			nstart[name] = self.OG.node[name]['normweightmax']
		
		h, a = nx.hits(self.OG, max_iter = 30)
		for node in self.OG.nodes():
			self.OG.node[node]['hub'] = h[node]
			self.OG.node[node]['authority'] = a[node]

		#for WOG
		nstart2 = {}
		for name in nx.nodes(self.WOG):
			nstart2[name] = self.WOG.node[name]['normweightmax']
		
		h2, a2 = nx.hits(self.WOG, max_iter = 30)
		for node in self.WOG.nodes():
			self.WOG.node[node]['hub'] = h2[node]
			self.WOG.node[node]['authority'] = a2[node]

	def __set_weekend_change(self):
		changelist = []
		for node in self.OG.nodes():
			self.OG.node[node]['hub_change']  = (self.WOG.node[node]['hub'] - self.OG.node[node]['hub']) / self.OG.node[node]['hub']
			self.OG.node[node]['aut_change']  = (self.WOG.node[node]['authority'] - self.OG.node[node]['authority']) / self.OG.node[node]['authority']


class GeospatialEffect(BaseEstimator):
	'''
	n x n matrix of distance for each node to every other node
	'''

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

class RegionEncoding():

	"""
	This class will encode the region as dummy variables.

	Returns np array of dummy variables
	"""
	def __init__(self, graph):
		self.OG = graph

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return self.one_hot_encode(self.OG)


	def one_hot_encode(self, graph):
		# generate the array/data frame first
		# encode using ext library
		templist=[]
		for node in graph.nodes():
			templist.append((graph.node[node]['region'], graph.node[node]['planning_area']))
		
		arealist = pd.DataFrame(templist, columns=['region','planning_area'])

		region_dummy = pd.get_dummies(arealist['region'], prefix='region', drop_first=True)
		pa_dummy = pd.get_dummies(arealist['planning_area'], prefix='pa', drop_first=True)

		arealist.drop(['region'], axis=1, inplace=True)
		arealist = arealist.join(region_dummy)
		arealist.drop(['planning_area'], axis=1, inplace=True)
		
		return np.array(arealist)



class BasicFeatureBuilder():

	"""
	This class builds the basic features of the graph if the graph does not have max_iter

	Returns np array of basic features
	"""
	
	def __init__(self, maingraph, originalgraph, breedinghabitat, build=True):
		self.G = maingraph
		self.OG = originalgraph
		self.BH = breedinghabitat
		#self.__generate_binary()
		if build:
			self.__build()


	def export_gexf(self, filename):
		nx.write_gexf(self.OG, filename)

	def __build(self):
		self.__centrality_analysis()
		self.__link_analysis()
		self.__bh_proximity_density()
		
		
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		x_list = []
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
						self.OG.node[area]['bh_count']
						))

		X = np.array(x_list)
		return X

	def get_y(self):
		y_list = []
		for area in self.OG.nodes():
			y_list.append(self.OG.node[area]['active_hotspot'])
		y = np.array(y_list)
		return y
			
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

	def __centrality_analysis(self):
		eigen_centrality = nx.eigenvector_centrality(self.OG, weight = 'normweightbymax')
		btw_centrality = nx.betweenness_centrality(self.OG, weight = 'normweightbymax')
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


class DeltaFeatureBuilder():
	'''
	This class builds the main features AND labels for predicting change in dengue case numbers.

	labels : uses the difference in #cases (stored in the first graph)

	features: subtracts end feature value from start feature value 

	Users will be able to use X day's of change to forecase the change in Y day's time

	'''

	def __init__(self, graphlist, change, ahead, pred_movement):
		self.inputlist = graphlist
		self.graphlist = self.__build(graphlist, change, ahead, pred_movement)
		self.change = change
		self.ahead = ahead
		self.pred_move = pred_movement

	def __build(self, graphlist, change, ahead, pred_move):
		newlist=[]
		i = 0
		while (i+change) < (len(graphlist) - ahead): 
			start = graphlist[i][1]
			end = graphlist[i+change][1]

			for node in start.nodes():
				start.node[node]['delta_BNI'] = end.node[node]['bad_neighbour_in'] - start.node[node]['bad_neighbour_in']
				start.node[node]['delta_BNO'] = end.node[node]['bad_neighbour_out'] - start.node[node]['bad_neighbour_out']
				start.node[node]['delta_BN2I'] = end.node[node]['bad_neighbour_in2'] - start.node[node]['bad_neighbour_in2']
				start.node[node]['delta_BN2O'] = end.node[node]['bad_neighbour_out2'] - start.node[node]['bad_neighbour_out2']
				start.node[node]['delta_EC'] = end.node[node]['eigen_centrality'] - start.node[node]['eigen_centrality']
				start.node[node]['delta_BC'] = end.node[node]['betweenness_centrality'] - start.node[node]['betweenness_centrality']
				start.node[node]['delta_pagerank'] = end.node[node]['pagerank'] - start.node[node]['pagerank']
				start.node[node]['delta_hub'] = end.node[node]['hub'] - start.node[node]['hub']
				start.node[node]['delta_authority'] = end.node[node]['authority'] - start.node[node]['authority']
				start.node[node]['delta_bh_density'] = end.node[node]['bh_density'] - start.node[node]['bh_density']
				start.node[node]['delta_bh_count'] = end.node[node]['bh_count'] - start.node[node]['bh_count']
				start.node[node]['delta_inverse_dist'] = end.node[node]['inverse_dist'] - start.node[node]['inverse_dist']
				start.node[node]['cases'] = end.node[node]['weight'] # not sure if supposed to be like this?? (predict from the last day to next day)
				if pred_move:
					start.node[node]['delta_cases'] = graphlist[i+change+ahead][1].node[node]['weight'] - end.node[node]['weight'] #test this tmr
				else:
					start.node[node]['delta_cases'] = graphlist[i+change+ahead][1].node[node]['weight']
				
			i+=1
			newlist.append(start)
			# contains graphs with relevant changes (n - number of days ahead) days worth of graph
		return newlist

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		x_list = []

		for i in range(len(self.graphlist)):
			temp = self.graphlist[i]
			for node in temp.nodes():
				x_list.append((temp.node[node]['delta_BNI'],\
								temp.node[node]['delta_BNO'],\
								temp.node[node]['delta_BN2I'],\
								temp.node[node]['delta_BN2O'],\
								temp.node[node]['delta_EC'],\
								temp.node[node]['delta_BC'],\
								temp.node[node]['delta_pagerank'],\
								temp.node[node]['delta_hub'],\
								temp.node[node]['delta_authority'],\
								temp.node[node]['delta_bh_density'],\
								temp.node[node]['delta_bh_count'],\
								temp.node[node]['delta_inverse_dist']\
								#temp.node[node]['cases']
								))

		X = np.array(x_list)
		return X


	def get_y(self):
		'''
		Creates labels based on change in number of cases
		
		'delta cases' depends on the boolean pred_movement
		true:    represents net change
		false:	 represents status
		'''
		y_list = []
		'''
		i=0
		while i < len(self.graphlist):
			temp = self.graphlist[i]
			for node in temp.nodes():
				if temp.node[node]['delta_cases'] > 0:
					y_list.append(1)
				#elif temp.node[node]['delta_cases'] < 0:
					#y_list.append(2)
				else:
					y_list.append(0)
			i+=1
		'''
		for temp in self.graphlist:
			for node in temp.nodes():
				#y_list.append(temp.node[node]['delta_cases'])
				
				if temp.node[node]['delta_cases'] > 0:
					y_list.append(1)
				else:
					y_list.append(0)
				
		y = np.array(y_list)
		return y


class DailyChange():
	'''
	day on day changes

	key terms:
	study period - number of days change u want to observe
	change = study period - 1
	ahead - number of days ahead to predict
	number of obs set = N - (study period + ahead) + 1
		e.g. 7 days of data, but with 3 days study period to predict 1 day ahead
		1  2  3  4  5  6  7
		x  x  X     p
		   x  x  X     p
		      x  x  X     p
		      	 
      	number of obs set = 7 - (3 + 2) + 1

	'''
	def __init__(self, graphlist, change, ahead):
		# graphlist contains only OG
		self.change = change
		self.ahead = ahead
		self.graphlist = graphlist
		self.newlist = self.calculate_change(graphlist)


	def calculate_change(self, graphlist):
		'''
		generate the change for everyday and store in the next day. e.g. Tues-Wed change is stored in Wed

		returns list of graph with extra node attributes (change from the day before), EXCEPT first graph
		'''
		list_= []
		for i, graphtup in enumerate(graphlist):
			graph = graphtup[1]
			if i: #not the first day and not 
				for node in graph.nodes():
					graph.node[node]['dc_BNI'] = graph.node[node]['bad_neighbour_in'] - graphlist[i-1][1].node[node]['bad_neighbour_in']
					graph.node[node]['dc_BNO'] = graph.node[node]['bad_neighbour_out'] - graphlist[i-1][1].node[node]['bad_neighbour_out']
					graph.node[node]['dc_BN2I'] = graph.node[node]['bad_neighbour_in2'] - graphlist[i-1][1].node[node]['bad_neighbour_in2']
					graph.node[node]['dc_BN2O'] = graph.node[node]['bad_neighbour_out2'] - graphlist[i-1][1].node[node]['bad_neighbour_out2']
					graph.node[node]['dc_EC'] = graph.node[node]['eigen_centrality'] - graphlist[i-1][1].node[node]['eigen_centrality']
					graph.node[node]['dc_BC'] = graph.node[node]['betweenness_centrality'] - graphlist[i-1][1].node[node]['betweenness_centrality']
					graph.node[node]['dc_PR'] = graph.node[node]['pagerank'] - graphlist[i-1][1].node[node]['pagerank']
					graph.node[node]['dc_hub'] = graph.node[node]['hub'] - graphlist[i-1][1].node[node]['hub']
					graph.node[node]['dc_aut'] = graph.node[node]['authority'] - graphlist[i-1][1].node[node]['authority']
					graph.node[node]['dc_bh_den'] = graph.node[node]['bh_density'] - graphlist[i-1][1].node[node]['bh_density']
					graph.node[node]['dc_bh_count'] = graph.node[node]['bh_count'] - graphlist[i-1][1].node[node]['bh_count']
					graph.node[node]['dc_inverse_dist'] = graph.node[node]['inverse_dist'] - graphlist[i-1][1].node[node]['inverse_dist']
					graph.node[node]['dc_case'] = graph.node[node]['weight'] - graphlist[i-1][1].node[node]['weight']
				list_.append(graph)
		return list_

	def get_feat(self):
		change = self.change
		ahead = self.ahead
		graphlist = self.newlist
		'''
		gets features for the day on day change in the study period
		stores features under last day of study period

		e.g. 3 day period for 7 day Data (c represents dc values while C represents the day 
			which stores the data for all change in study period)
		3 day study period means change = 2 
		1 2 3 4 5 6 7
		  c C  
		  	c C
		  	  c C
		  	    c C
		  	      c C

		'''

		#put graphlist data into list, hstack the list then vstack the days

		number_of_sets = len(graphlist) - change - ahead + 1  #IMPT formula
		feature=[]
		feat=[]
		for i in range(number_of_sets):
			# each day, collect set n hstack
			periodlist = []
			for j in range(change):
				daylist=[]
				for node in graphlist[i+j].nodes():
					temp = graphlist[i+j]
					daylist.append((temp.node[node]['dc_BNI'],\
									temp.node[node]['dc_BNO'],\
									temp.node[node]['dc_BN2I'],\
									temp.node[node]['dc_BN2O'],\
									temp.node[node]['dc_EC'],\
									temp.node[node]['dc_BC'],\
									temp.node[node]['dc_PR'],\
									temp.node[node]['dc_hub'],\
									temp.node[node]['dc_aut'],\
									temp.node[node]['dc_bh_den'],\
									temp.node[node]['dc_bh_count'],\
									temp.node[node]['dc_inverse_dist'],\
									temp.node[node]['dc_case']
									))
				periodlist.append(daylist)

			feat=np.hstack(periodlist)
			feature.append(feat)

		feature=np.vstack(feature)
		
		print feature.shape

		return feature


class MovingAverage():
	def __init__(self, graphlist, daycount):
		self.graphlist = graphlist
		self.day = daycount







if __name__ == '__main__':
	network_data = pd.read_csv("Data/network_20160511.csv")
	wkend_network_data = pd.read_csv("Data/Original/20160514_network.csv")
	subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")

	GG = GraphGenerator(network_data, subzone_data)
	GG2 = GraphGenerator(wkend_network_data, subzone_data)
	G, OG, BH = GG.get_graphs()
	WG, WOG, WBH = GG2.get_graphs()
	x = []

	BFB = BasicFeatureBuilder(G, OG, BH)
	BN = BadNeighbours(OG, BH)
	#X = BN.fit(x).transform(x)
	#FB = BasicFeatureBuilder(G, OG, BH)
	#FB2 = BasicFeatureBuilder(WG, WOG, WBH)

	HC = HitsChange(OG, WOG)
	#X1 = HC.fit(x).transform(x)

	y=[]

	FU = FeatureUnion([('fb', BFB), ('bn',BN),('hc',HC)])
	F = FU.fit_transform(x,y)

	print F
	print len(F)
	print F.shape()
	
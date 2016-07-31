import os.path
import csv
import numpy as np
import pandas as pd
import networkx as nx

from shape import BasicExtractor, BreedingHabitatExtractor, DengueCaseExtractor, SubzoneShapeExtractor
from graph import GraphGenerator, FastGraphGenerator
from features import BasicFeatureBuilder, GeospatialEffect, HitsChange, BadNeighbours, DeltaFeatureBuilder, DailyChange, RegionEncoding
from sklearn.pipeline import FeatureUnion

"""
Util.py's only job is to function as a utility file
0. takes in list of dates
1. process graphs
	1. unpack shapefile
	2. unpack CSV files
	3. build normal features

build phase 2 features (change in time ones)
stack

2. make labels
3. return X and y to train.py
"""



network = "Data/Original/network/20160523.csv"
population = "Data/Original/population.csv"
shape = "shape/subzone.shp"
area = "shape/area.shp"
region = "shape/region.shp"
#shape = "shape/WSG84_GTAreas_1Apr2016/WSG84_GTAreas_1Apr2016.shp"

def load_multiple_data(datelist, study_period=2, ahead=1, delta=True, pred_movement=True, daily=False, finer_subzone=False):
	"""this method will take in a list of dates and return feature and labels as numpy array
	1. opens list of dates' corresponding breedinghabitat and dengue case files
	2. make list of graphs
	3. build features
	4. extract features into numpy array

	Parameters
	----------
	datelist : list of string
		list of dates in yyyymmdd format
	
	study_period : int (default = 2)
		minimum 2. length of observation period 
	
	ahead : int (default = 1)
		minimum 1. numbers of days ahead to predict
	
	delta : boolean, default:True
		determines if delta-features are to be included
	
	pred_movement : boolean, default:True
		determines if prediction labels will be generated
	
	daily : boolean, default:True
		determines if day on day changes will be included
	
	finer_subzone : boolean, default:True
		determines resolution of subzone to be used

	Returns 
	-------
	features : numpy array
		n-by-m array of features (m is number of features)

	labels : numpy array
		n-by-1 array of labels (1 and 0)

	"""
	days_change = study_period-1 # equals to study period - 1
	days_ahead = ahead
	graphlist = []
	for date in datelist:
		graphlist.append((date,load_one_day_data(date, finer_subzone)) )
		print "unpacked " + date
	
	feat = []
	lab = []
	updated_graphlist = []

	i=0
	while i < len(graphlist):
		if finer_subzone:
			graphfilename = "Data/Processed/Storage/" + graphlist[i][0] + "graph.csv"
		else: 
			graphfilename = "Data/Processed/Storage_subzone/" + graphlist[i][0] + "graph.csv" 

		tup = graphlist[i][1]
		
		if os.path.exists(graphfilename):
			graphDF = pd.read_csv(graphfilename)
			GG = FastGraphGenerator(graphDF)
			OG = GG.get_graph(finer_subzone)
			X, OG, BH = basic_feature_selecter(tup[0], OG, tup[2], False, finer_subzone)
		else:
			print "nope, tough luck"
			X, OG, BH = basic_feature_selecter(tup[0], tup[1], tup[2], finer = finer_subzone)
			save_graph(graphfilename,OG, finer_subzone)
		if i < (len(graphlist) - days_change - days_ahead):
			feat.append(X)
			
		updated_graphlist.append((tup[0], OG, BH))
		print "Built basic features for day %d" % (i+1)
		i+=1

	basic_feat = np.vstack(feat)
	
	features = []
	features.append(basic_feat)
	
	delta_feat, labels = delta_feature_selector(updated_graphlist, days_change, days_ahead, pred_movement)
	
	if delta:
		features.append(delta_feat)
	
	# add in daily change
	if daily:
		DC = DailyChange(updated_graphlist, days_change, days_ahead)
		X = DC.get_feat()
		features.append(X)
	
	features = np.hstack(features)

	return features, labels

def save_graph(filename, graph, finer_subzone):
	"""Saves node attributes in a csv file

	Parameters
	----------
	filename : string
		name of file to be saved as

	graph : networkx Digraph
		input graph to be converted into csv

	finer_subzone : boolean
		indicate resolution of subzone
		finer subzones do not have population related attributes	
	"""

	x_list=[]

	if not finer_subzone:
		for area in graph.nodes():
			x_list.append((area,
					graph.node[area]['weight'], graph.node[area]['normweightmax'],\
					graph.node[area]['longitude'], graph.node[area]['latitude'],\
					graph.node[area]['area'], graph.node[area]['population'],\
					graph.node[area]['popdensity'], graph.node[area]['bh_count'],\
					graph.node[area]['eigen_centrality'], graph.node[area]['betweenness_centrality'],\
					graph.node[area]['pagerank'], graph.node[area]['hub'],\
					graph.node[area]['authority'], graph.node[area]['bh_density'],\
					graph.node[area]['inverse_dist'],graph.node[area]['bad_neighbour_in'],\
					graph.node[area]['bad_neighbour_out'], graph.node[area]['bad_neighbour_in2'],\
					graph.node[area]['bad_neighbour_out2'], graph.node[area]['region'],\
					graph.node[area]['planning_area'], graph.node[area]['clustering']))
			
			column = ['Subzone','weight','Cases_Norm_Max','Lon','Lat','Area','Population','Pop_density',\
					'BH_count' ,'EC', 'BC','PR', 'hub', 'aut', 'bh_density','inverse_dist', 'bni',\
					'bno','bn2i','bn2o', 'region', 'planning_area', 'clustering']
	else:
		for area in graph.nodes():
			x_list.append((area,
					graph.node[area]['weight'], graph.node[area]['normweightmax'],\
					graph.node[area]['longitude'], graph.node[area]['latitude'],\
					graph.node[area]['area'], graph.node[area]['bh_count'],\
					graph.node[area]['eigen_centrality'], graph.node[area]['betweenness_centrality'],\
					graph.node[area]['pagerank'], graph.node[area]['hub'],\
					graph.node[area]['authority'], graph.node[area]['bh_density'],\
					graph.node[area]['inverse_dist'],graph.node[area]['bad_neighbour_in'],\
					graph.node[area]['bad_neighbour_out'], graph.node[area]['bad_neighbour_in2'],\
					graph.node[area]['bad_neighbour_out2'], graph.node[area]['region'],\
					graph.node[area]['planning_area'], graph.node[area]['clustering']))
			
			column = ['Subzone','weight','Cases_Norm_Max','Lon','Lat','Area',\
					'BH_count' ,'EC', 'BC','PR', 'hub', 'aut', 'bh_density','inverse_dist', 'bni',\
					'bno','bn2i','bn2o', 'region', 'planning_area', 'clustering']

	frame = pd.DataFrame(x_list, columns = column)
	frame.to_csv(filename, index=False)


def basic_feature_selecter(G, OG, BH, build=True, finer=True):
	"""Uses FeatureUnion to fit basic feature builders
	Parameters
	----------
	G : networkx Digraph
		entire graph networkx (included for future flexibility, for gexf file creation)

	OG : networkx Digraph
		main graph to be processed

	BH : networkx Graph
		breeding habitat graph

	build : boolean, default: True
		indicate to featurebuilders whether to construct functions again
		if graphs are loaded, features would have been built in previous runs. This reduces
		runtime.

	finer : boolean
		indicate resolution of subzones
	
	Returns
	-------
	X : numpy array
		feature array

	NOG : networkx Digraph
		post processed original graph 

	NBH : networkx graph
		post processed breedinghabitat
	"""
	X = [] # empty for feature union
	y = [] # same
	BFB = BasicFeatureBuilder(G, OG, BH, build, finer)
	BN = BadNeighbours(BFB.OG, BFB.BH, tobuild=build, second_degree=True)
	RE = RegionEncoding(OG)
	FU = FeatureUnion([('basic', BFB), ('neighbours',BN), ('region', RE)])
	X = FU.fit_transform(X,y)

	NOG = BN.OG
	NBH = BFB.BH
	return X, NOG, NBH

def delta_feature_selector(graphlist, change, ahead, pred):
	"""Builds delta features

	Parameters
	----------
	graphlist : list of networkx Digraph
		sorted list of graphs 

	change : int
		indicate number of days change to calculate

	ahead : int
		number of days ahead to predict

	pred : boolean
		indicate type of labels to build
		True: based on movement
		False: based on status

	Returns
	-------
	X : numpy array
		feature array of delta features only

	y : numpy array
		array of labels
	"""
	DF = DeltaFeatureBuilder(graphlist, change, ahead, pred)
	y = DF.get_y()
	X = []  # can actually edit feature selector
	X = DF.fit(X, y).transform(X)
	return X, y

def make_file_path(date, finer_subzone):
	"""Generates various filepath based on dates

	Parameters
	----------
	date : string
		date in yyyymmdd format

	finer_subzone : boolean
		indicates resolution

	Returns
	-------
	caselist : list of string
		dengue cases file path for 5 regions

	bhlist : list of string
		breedinghabitat file path for 5 regions

	network : string
		network traffic file path
	"""
	date = str(date)
	bhlist = ['Data/Original/DailyData/' + date + '/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
				'Data/Original/DailyData/' + date + '/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
				'Data/Original/DailyData/' + date + '/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
				'Data/Original/DailyData/' + date + '/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
				'Data/Original/DailyData/' + date + '/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp']

	caselist = ['Data/Original/DailyData/' + date + '/denguecase-central-area/DengueCase_Central_Area.shp',\
				'Data/Original/DailyData/' + date + '/denguecase-northeast-area/DengueCase_Northeast_Area.shp',\
				'Data/Original/DailyData/' + date + '/denguecase-northwest-area/DengueCase_Northwest_Area.shp',\
				'Data/Original/DailyData/' + date + '/denguecase-southeast-area/DengueCase_Southeast_Area.shp',\
				'Data/Original/DailyData/' + date + '/denguecase-southwest-area/DengueCase_Southwest_Area.shp']
	
	if finer_subzone:
		network = "Data/Original/zone_network/" + date + ".csv"
	else:
		network = "Data/Original/network/" + date + ".csv"
	return caselist, bhlist, network

def load_one_day_data(date, finer_subzone): # loading one day data
	"""Generates 3 main graphs for 1 day of Data

	Parameters
	----------
	date : string
		date in yyyymmdd format

	finer_subzone : boolean
		indicates resolution

	Returns
	-------
	graph_tuple : tuple of networkx graphs
		contains full graph, nodes-only graph and breedinghabitat
	"""
	
	if finer_subzone:
		nodefilename="Data/Processed/Storage/" + date + "node.csv"
		edgefilename="Data/Processed/Storage/" + date + "edge.csv"
		bhfilename="Data/Processed/Storage/" + date + "bh.csv"
	else:
		nodefilename="Data/Processed/Storage_subzone/" + date + "node.csv"
		edgefilename="Data/Processed/Storage_subzone/" + date + "edge.csv"
		bhfilename="Data/Processed/Storage_subzone/" + date + "bh.csv"

	if os.path.exists(nodefilename):
		# if file is in storage, open it without preprocessing
		#print "file exists, youre in luck"
		nodeDF=pd.read_csv(nodefilename)
		edgeDF=pd.read_csv(edgefilename)
		bhDF=pd.read_csv(bhfilename)
	else:
		caselist, bhlist, network = make_file_path(date, finer_subzone)
		SSE = SubzoneShapeExtractor(shape, area, region)
		BHE = BreedingHabitatExtractor(bhlist, shape)
		DCE = DengueCaseExtractor(caselist, shape)

		node_geospatial_details = SSE.get_list() # subzone, lon, lat, area
		node_case_details = DCE.get_list()       # subzone, case_count
		node_bh_count = BHE.get_loclist()        # subzone, bh_count, lon, lat
		bh_lon_lat = BHE.get_fulllist()			 # lon, lat

		nodeDF = pd.DataFrame(node_geospatial_details, columns=['Subzone', 'Lon', 'Lat', 'Area', 'Planning_Area', 'Region'])
		index = pd.Series(nodeDF['Subzone'])
		nodeDF = nodeDF.set_index(index)

		bhDF = pd.DataFrame(node_bh_count, columns=['subzone','BH_count'])
		index2 = pd.Series(bhDF['subzone'])
		bhDF = bhDF.set_index(index2)
		#nodeDF = nodeDF.sort_index()

		caseDF = pd.DataFrame(node_case_details, columns =['subzone', 'Cases'])
		index3 = pd.Series(caseDF['subzone'])
		caseDF = caseDF.set_index(index3)

		if not finer_subzone:
			popDF = pd.read_csv(population) # subzone, population count
			index = index.sort_values()
			popDF = popDF.set_index(index)
			nodeDF = pd.concat([nodeDF, bhDF, popDF, caseDF], axis=1)
			nodeDF['Pop_density'] = (nodeDF['Population']/nodeDF['Area']).astype(float)
		else: 
			nodeDF = pd.concat([nodeDF, bhDF, caseDF], axis=1)
		
		nodeDF = nodeDF.fillna(0.0)
		
		nodeDF['Cases_Norm_Max'] = ((nodeDF['Cases'] - min(nodeDF['Cases']))/(max(nodeDF['Cases'] - min(nodeDF['Cases']))))

		bhDF = pd.DataFrame(bh_lon_lat, columns = ['Lon', 'Lat'])
		edgeDF = pd.read_csv(network, names=['Source', 'Target', 'Weight'])

		#write_to_csv

		nodeDF.to_csv(nodefilename, index=False)
		edgeDF.to_csv(edgefilename, index=False)
		bhDF.to_csv(bhfilename, index=False)

	GG = GraphGenerator(nodeDF, edgeDF, bhDF, finer=finer_subzone)
	G, OG, BH = GG.get_graphs()

	return (G, OG, BH)
	

if __name__ == '__main__':
	list_ = ["20160523", "20160524", "20160525", "20160527", "20160531", "20160601", "20160602"]
	x, y = load_multiple_data(list_)



"""
cluster = "DailyData/240516/dengue-clusters/DENGUE_CLUSTER.shp"

bhlist = ['Data/Original/DailyData/20160523/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
			'Data/Original/DailyData/20160523/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
			 'Data/Original/DailyData/20160523/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
			 'Data/Original/DailyData/20160523/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
			 'Data/Original/DailyData/20160523/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp',]

caselist = ['Data/Original/DailyData/20160523/breedinghabitat-central-area/BreedingHabitat_Central_Area.shp',\
			'Data/Original/DailyData/20160523/breedinghabitat-northeast-area/BreedingHabitat_Northeast_Area.shp',\
			'Data/Original/DailyData/20160523/breedinghabitat-northwest-area/BreedingHabitat_Northwest_Area.shp',\
			'Data/Original/DailyData/20160523/breedinghabitat-southeast-area/BreedingHabitat_Southeast_Area.shp',\
			'Data/Original/DailyData/20160523/breedinghabitat-southwest-area/BreedingHabitat_Southwest_Area.shp']
"""

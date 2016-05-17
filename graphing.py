import numpy as np
import pandas as pd
import networkx as nx
import itertools


# opening files
network_data = pd.read_csv("Data/network_20160511.csv")
subzone_data = pd.read_csv("Data/subzone_cases_20160511.csv")

DG = nx.DiGraph()

transformer_dict = {}

# number the nodes
def transform_feature(df, df2, column_name):
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

def make_graph(nodes, network):
	make_edge(network)
	make_node(nodes)

# input direction into graph
def make_edge(network):
	
	subset = network[["Source", "Target", "Weight"]]
	edge_list = [tuple(x) for x in subset.values]
	DG.add_weighted_edges_from(edge_list)
	
	#DG = nx.from_pandas_dataframe(network, "Source", "Target", ['Weight'])
	
def make_node(nodes):
	subset = nodes["subzone"]
	weight_set = nodes["cases"]
	
	for i, subzone in enumerate(subset):
		DG.add_node(subzone, weight = weight_set[i])


transform_feature(subzone_data, network_data, "subzone")

make_graph(subzone_data, network_data)


#print nx.number_of_nodes(DG)
#print nx.nodes(DG)
#print DG.node
print nx.info(DG)
print nx.all_neighbors(DG, 1)


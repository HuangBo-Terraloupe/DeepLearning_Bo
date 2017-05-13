import json
import os
import sys

import geopandas as gps
import networkx as nx
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from scipy.spatial import distance_matrix
from shapely.geometry.linestring import LineString
from objectdetection.utils.preprocessing_utils import get_metadata, get_source_feature_ids

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils import logger


class LineMerger(BasePreprocessor):
    '''
        Merges various line features to one line if the distances between them is below a threshold. 
    '''
    
    class Node:
        '''
        A node to represent one end of the line.
        Parameters:
            feature_id: id of the feature to which this line belongs.
            is_start: flag to mark the starting point of the line.
        '''
        
        def __init__(self, is_start, feature_id):
            self.is_start = is_start
            self.feature_id = feature_id
        
        def __repr__(self):
            return "Node :%d(%d)" % (self.feature_id, self.is_start) 
    
    
    def __init__(self, input_dir, outdir, **kwargs):
        data_category = kwargs['category']
        threshold = kwargs['threshold']
        self.META_DATA = get_metadata(self.__class__.__name__, data_category)
        out_data_category = data_category.replace(' ', '_')
        filename = '{}-LineString.geojson'.format(out_data_category)
        input_file_path = os.path.join(input_dir, filename)
        self.input_data = gps.read_file(input_file_path)
        self.target_geojson_filename = os.path.join(outdir, filename)
        self.category_data = self.input_data[self.input_data.category == data_category]
        self.category_data = self.category_data[self.input_data.geometry.type == 'LineString']
        self.data_category = data_category 
        self.graph = nx.Graph()
        self.feature_map = {}
        self.nodes = []
        self.threshold = threshold
    
    def build_graph(self):
        logger.info('Building graph for lines')
        points = []
        for (_, feature) in self.category_data.iterrows():
            feature_id = int(feature.id)
            self.feature_map[feature_id] = feature
            line = feature.geometry
            coords = list(line.coords)
            points.append(coords[0])
            points.append(coords[-1])
            
            n1 = LineMerger.Node(True, feature_id)
            n2 = LineMerger.Node(False, feature_id)
            self.nodes.append(n1)
            self.nodes.append(n2)
            self.graph.add_edge(n1, n2, {"internal": True})
        self.points = np.array(points)
        logger.info('Built graph for all lines')
    
    def join_nearest_neighbours(self):
        logger.info('Joining nearest neighbours with distance less than {}'.format(self.threshold))
        dm = distance_matrix(self.points, self.points)
        for idx, node in enumerate(self.nodes):
            if self.graph.degree(node) > 1:
                logger.debug('Node:{} is already merged with other node, skipping search for nearest node.'.format(node))
                continue
            min_dist = sys.maxint
            best_idx = -1
            for other_idx, d in enumerate(dm[idx]):
                if idx != other_idx and d < self.threshold and d < min_dist:
                    if self.nodes[other_idx] not in self.graph.edge[node]:
                        min_dist = d
                        best_idx = other_idx

            # don't create an edge if the nearest node is already joined,
            if best_idx != -1 and self.graph.degree(self.nodes[best_idx]) < 2:
                self.graph.add_edge(node, self.nodes[best_idx], {'internal': False})
    
    def get_start_node(self, graph):
        '''
         A strongly connected graph, can be  
         1. A long line, all nodes except end points of line will have degree 2, end points will have degree 1
         2. A loop, all nodes will have degree 2.
         This method picks one of two end nodes in case 1 and picks the first node from the node list for 2nd case.
        '''
        node_degrees = graph.degree()
        start_node = None
        is_loop = False
        for node, degree in node_degrees.iteritems():
            if degree == 1:
                start_node = node
                break
        
        # Connected nodes form a loop, pick any node as start node.
        if start_node is None:
            start_node = graph.nodes()[0]
            is_loop = True
        return start_node, is_loop
    
    class Line:
        def __init__(self, nodeA, nodeB, coords):
            self.nodeA = nodeA
            self.nodeB = nodeB
            self.coords = coords
            # reverse the co-ordinates so that all lines orient in one direction only
            if not self.nodeA.is_start:
                self.coords.reverse()
                
        def __repr__(self):
            return 'Line :{}, {}->{}'.format(self.nodeA.feature_id, self.nodeA.is_start, self.nodeB.is_start)
    
    def merge_feature_collection(self):
        logger.info('Merging multiple connected lines into one feature.')
        new_features = []
        source_feature_map = {}
        for i, subgraph in enumerate(list(nx.connected_component_subgraphs(self.graph))):
            source_feature_ids = set()
            start_node, is_loop = self.get_start_node(subgraph)
            
            # add visited attribute for traversal ...  
            map(lambda x: setattr(x, 'visited', False), subgraph.nodes())
            nodeA, last_node = start_node, None
            all_coords = list(self.feature_map[nodeA.feature_id].geometry.coords)
            lines = []
            while nodeA and not nodeA.visited:
                neighbours = subgraph.neighbors(nodeA)
                nodeB = None
                for neighbour in neighbours:
                    if neighbour != last_node:
                        nodeB = neighbour
                
                if nodeB and nodeA.feature_id == nodeB.feature_id:
                    line = LineMerger.Line(nodeA, nodeB, list(self.feature_map[nodeA.feature_id].geometry.coords))
                    lines.append(line)
                
                last_node = nodeA
                nodeA.visited = True
                nodeA = nodeB
            
            all_coords = []
            for line in lines:
                all_coords.extend(line.coords)
                source_feature_ids.add(line.nodeA.feature_id)
                source_feature_ids.add(line.nodeB.feature_id)
            
            # if a loop is formed, append the first co-ordinate at the end ... 
            if is_loop and np.linalg.norm(np.array(all_coords[0]) - np.array(all_coords[-1])) != 0:
                all_coords.append(all_coords[0])
            
            source_feature_map[i] = source_feature_ids
            line_string = LineString(np.array(all_coords))
            new_features.append(line_string)
        
        new_geo_series_list = []
        for i, feature in enumerate(new_features):
            source_features = [self.feature_map[feature_id] for feature_id in source_feature_map[i]]
            source_feature_ids = get_source_feature_ids(source_features)
            data = {'geometry' : feature, 'source_features' : ','.join(source_feature_ids), 'id' : str(i)}
            data.update(self.META_DATA)
            new_geo_series_list.append(GeoSeries(data))
            
        self.geo_data = GeoDataFrame(new_geo_series_list)
    
    def save_geo_json(self):
        geo_json = self.geo_data.to_json()
        parsed_geo_json = json.loads(geo_json)
        with open(self.target_geojson_filename, 'w') as fw:
            fw.write(json.dumps(parsed_geo_json, indent=4, sort_keys=True))
        logger.info('Successfully wrote geojson for merged lines to file:{}'.format(self.target_geojson_filename))
    
    
    def merge_lines(self):
        '''
        Merges the close lines into a single by doing following steps.
        1. Builds graph for each line in the input data. Each road edge feature is represented by two nodes and 
            these nodes have an edge between them with label 'internal' set as True. 
        2. Joins the graph components with the nearest graphs component with distance less than threshold.
        3. Merges the different connected components into one single line.
        4. Creates Geo json for all lines and saves to a file. 
        
        '''
        self.build_graph()
        self.join_nearest_neighbours()
        self.merge_feature_collection()
        self.save_geo_json()
    
    def process_data(self):
        self.merge_lines()
    
    def supported_geometries(self):
        return set(['LineString'])

# if __name__ == '__main__':
#     line_merger = LineMerger('/home/sanjeev/debug/input.geojson',
#                              '/home/sanjeev/debug/',
#                              category='road edge', threshold=0.3)
#     line_merger.process_data()

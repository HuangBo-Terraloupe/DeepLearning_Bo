import sys

import networkx as nx
import numpy as np
from scipy.spatial.kdtree import distance_matrix
from shapely.geometry.linestring import LineString
from objectdetection.utils import logger

from objectdetection.utils.time_utils import time_it


class LineConnector:
    '''
    Finds the longest continuous line from a list of list of lines.
    '''
    
    class Node:
        def __init__(self, uid, point, is_start):
            self.uid = uid 
            self.point = point
            self.is_start = is_start
        
        def __repr__(self):
            return '{}, {}-{}'.format(self.uid, self.point, self.is_start)
    
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
            
    def build_graph(self, lines):
        logger.info('Building graph for lines')
        points = []
        self.nodes = []
        self.graph = nx.Graph()
        self.line_map = {}
        for i, line in enumerate(lines):
            coords = list(line.coords)
            points.append(coords[0])
            points.append(coords[-1])
            self.line_map[i] = line
            node1 = LineConnector.Node(i, coords[0], True)
            node2 = LineConnector.Node(i, coords[-1], False)
            self.graph.add_edge(node1, node2, {'internal':True, 'length':line.length})
            self.nodes.append(node1)
            self.nodes.append(node2)
        self.points = np.array(points)
        logger.info('Built graph for all lines')
    
    def join_nearest_neighbours(self, lines, threshold):
        self.build_graph(lines)
        logger.info('Joining nearest neighbours with distance less than {}'.format(threshold))
        dm = distance_matrix(self.points, self.points)
        for idx, node in enumerate(self.nodes):
            min_dist = sys.maxint
            best_idx = -1
            for other_idx, d in enumerate(dm[idx]):
                if idx != other_idx and d < threshold and d < min_dist:
                    if self.nodes[other_idx] not in self.graph.edge[node]:
                        min_dist = d
                        best_idx = other_idx
            
            # don't create an edge if the nearest node is already joined,
            if best_idx != -1 and self.graph.degree(self.nodes[best_idx]) < 3 and self.graph.degree(node) < 3:
                self.graph.add_edge(node, self.nodes[best_idx], {'internal': False, 'length': min_dist})
            
    def longest_path(self):
        subgraphs = list(nx.connected_component_subgraphs(self.graph))
        component_length = lambda subgraph: reduce(lambda x, y : x + y[2]['length'], subgraph.edges(data=True), 0)
        graph = max(subgraphs, key=component_length)
        if len(graph.nodes()) <= 2:
            return graph.edges(data=True)
        start_nodes = self.get_start_nodes(graph)
        best_path_length = -1
        best_path = None
        for start_node in start_nodes:
            map(lambda x: setattr(x, 'visited', False), graph.nodes())
            start_node.visited = True
            edges = graph.edges(start_node, data=True)
            longest_paths = []
            for edge in edges:
                longest_path = [edge] + self.find_longest_path(graph, edges[0][1])
                longest_paths.append(longest_path)
            
            longest_paths.sort(key=lambda x : self.path_length(x), reverse=True)
            longest_paths = longest_paths[:2] if len(longest_paths) > 2 else longest_paths
            final_path = [edge for path in longest_paths for edge in path]

            final_path_length = self.path_length(final_path)
            if final_path_length > best_path_length:
                best_path = final_path
                best_path_length = final_path_length
        return best_path
        
    def find_longest_path(self, graph, start_node):
        start_node.visited = True
        all_edges = filter(lambda x: not x[1].visited, graph.edges(start_node, data=True))
        if len(all_edges) == 0:
            return []
        else:
            best_path = None
            max_path_length = -1
            for edge in all_edges:
                path = self.find_longest_path(graph, edge[1])
                if edge[2]['length'] + self.path_length(path) > max_path_length:
                    best_path = path + [edge]
                    max_path_length = edge[2]['length'] + self.path_length(path)
            return best_path
    
    def get_start_nodes(self, graph):
        return graph.nodes()
    
    @time_it
    def get_longest_continuous_line(self, lines):
        '''
        The longest continuous line is found by
        1. Building a graph of end points of lines.
        2. Joining points which have distance less than threshold.
        3. Selecting the connected component with largest length.
        4. Finding the longest path in the  graph.
        5. Ordering all lines to form a continuous line.
        '''
        self.join_nearest_neighbours(lines, 0.05)
        longest_path = self.longest_path()
        new_graph = nx.Graph()
        new_graph.add_edges_from(longest_path)
        
        map(lambda x: setattr(x, 'visited', False), new_graph.nodes())
        start_nodes = [node for node, degree in new_graph.degree().iteritems() if degree == 1]
        nodeA, last_node = start_nodes[0], None
        lines = []
        while nodeA and not nodeA.visited:
            neighbours = new_graph.neighbors(nodeA)
            nodeB = None
            for neighbour in neighbours:
                if neighbour != last_node:
                    nodeB = neighbour
            
            if nodeB and nodeA.uid == nodeB.uid:
                line = LineConnector.Line(nodeA, nodeB, list(self.line_map[nodeA.uid].coords))
                lines.append(line)
            
            last_node = nodeA
            nodeA.visited = True
            nodeA = nodeB
        
        all_coords = []
        for line in lines:
            all_coords.extend(line.coords)
            
        return LineString(all_coords)
    
    def path_length(self, edges):
        return reduce(lambda x, y : x + y[2]['length'], edges, 0)
    

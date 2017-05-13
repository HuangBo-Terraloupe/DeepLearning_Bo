import os

import geopandas as gps
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from networkx.utils.union_find import UnionFind
from objectdetection.utils import logger, shape_utils
from objectdetection.utils.preprocessing_utils import  get_metadata, get_source_feature_ids

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils.io import gps_io_utils


class DuplicateLineDetector(BasePreprocessor):
    '''
    Detects duplicate lines in a geojson and writes the detected features in three separate files.
    1. The file which contains the line features which are marked as duplicate. Only line with high points is chosen.
    2. The file which contains intersecting lines, these can be manually inspected.
    3. The file which contains the remaining features not covered in above two files.
    '''
    
    def __init__(self, input_dir, out_folder_path, **kwargs):
        data_category = kwargs['category']
        boundary_length = kwargs['boundary_length']
        threshold = kwargs['threshold']
        self.META_DATA = get_metadata(self.__class__.__name__, data_category)
        out_data_category = data_category.replace(' ', '_')
        
        input_file_path = os.path.join(input_dir, '{}-LineString.geojson'.format(out_data_category))
        self.input_data = gps.read_file(input_file_path)
        self.data_category = data_category
        self.shape_data = self.input_data[self.input_data.geometry.type == 'LineString']
        self.category_data = self.shape_data[self.shape_data.category == data_category]
        self.boundary_length = boundary_length
        
        self.features = [feature for _, feature in self.category_data.iterrows()]
        self.feature_map = {feature.id: feature for feature in self.features}
        self.threshold = threshold
        # Out filenames...
        self.duplicate_feature_filename = os.path.join(out_folder_path, '{}-LineString-duplicate.geojson'.format(out_data_category))
        self.possible_duplicate_filename = os.path.join(out_folder_path, '{}-LineString-possible_duplicate.geojson'.
                                                    format(out_data_category))
        self.other_feature_filename = os.path.join(out_folder_path, '{}-LineString-other.geojson'.format(out_data_category))
        self.all_feature_filename = os.path.join(out_folder_path, '{}-LineString.geojson'.format(out_data_category))

    def save_duplicate_features(self, duplicate_feat_set):
        dup_feature_map = shape_utils.get_disjoin_sets(duplicate_feat_set)
        duplicate_features_gs = []
        for i, item in enumerate(dup_feature_map.iteritems()):
            _, duplicate_feature_ids = item
            best_duplicate_feature = self.feature_map[duplicate_feature_ids[0]]
            max_points = len(list(best_duplicate_feature.geometry.coords))
            for duplicate_feature_id in duplicate_feature_ids:
                duplicate_feature = self.feature_map[duplicate_feature_id]
                if max_points > len(list(duplicate_feature.geometry.coords)):
                    best_duplicate_feature = duplicate_feature
            source_features = [self.feature_map[feature_id] for feature_id in duplicate_feature_ids]
            source_feature_ids = get_source_feature_ids(source_features)
            data = {'geometry' : best_duplicate_feature.geometry, 'source_features' : ','.join(source_feature_ids),
                     'id' : str(i)}
            data.update(self.META_DATA)
            duplicate_features_gs.append(GeoSeries(data))
        
        duplicate_features_gdf = GeoDataFrame(duplicate_features_gs)
        gps_io_utils.write_gdf_json(duplicate_features_gdf, self.duplicate_feature_filename)
        return duplicate_features_gs

    def save_possible_duplicates(self, possible_duplicate_set):
        possible_dup_feature_map = shape_utils.get_disjoin_sets(possible_duplicate_set)
        possible_dup_features_gs = []
        i = 0 
        for _, feature_ids in possible_dup_feature_map.items():
            for feature_id in feature_ids:
                feature = self.feature_map[feature_id]
                source_feature_ids = get_source_feature_ids([feature])
                data = {'geometry' : feature.geometry, 'source_features' : ','.join(source_feature_ids), 'id' : str(i)}
                data.update(self.META_DATA)
                possible_dup_features_gs.append(GeoSeries(data))
                i += 1
        
        possible_dup_features_gdf = GeoDataFrame(possible_dup_features_gs)
        gps_io_utils.write_gdf_json(possible_dup_features_gdf, self.possible_duplicate_filename)
        return possible_dup_features_gs


    def save_remaining_features(self, possible_duplicate_set, duplicate_feat_set):
        all_feature_ids = set(map(lambda x : x.id, self.features))
        duplicate_feature_ids = set(duplicate_feat_set)
        possible_duplicate_feature_ids = set(possible_duplicate_set)
        other_feature_ids = all_feature_ids.difference(duplicate_feature_ids.union(possible_duplicate_feature_ids))
        logger.info('Total features:{}, detected duplicate features:{}, possible duplicate features:{}, other features:{}'.
                    format(len(self.features), len(duplicate_feature_ids), len(possible_duplicate_feature_ids), len(other_feature_ids)))
        other_features_gs = []
        for i, other_feature_id in enumerate(other_feature_ids):
            other_feature = self.feature_map[other_feature_id]
            source_feature_ids = get_source_feature_ids([other_feature])
            data = {'geometry' : other_feature.geometry, 'source_features' : ','.join(source_feature_ids), 'id' : str(i)}
            data.update(self.META_DATA)
            other_features_gs.append(GeoSeries(data))
        
        other_features_gdf = GeoDataFrame(other_features_gs)
        gps_io_utils.write_gdf_json(other_features_gdf, self.other_feature_filename)
        return other_features_gs

    def merge_duplicate_lines(self):
        '''
        Remove duplicate and lines and replaces with most precise line. Duplicate detection works using following steps  
        1. Create a polygon buffer around each line. 
        2. For each line check if it is entirely contained in the polygon buffer of any other line and vice versa. If yes
         then  the two lines are duplicates.
        3. If polygon buffers for lines intersect have a common area above threshold then they are possible duplicates and
         should be manually examined. 
        '''
        neighbouring_feature_map = shape_utils.get_neighbouring_shapes(self.features, self.boundary_length)
        buffer_polygon_map = {feature.id:feature.geometry.buffer(self.boundary_length,
                                    resolution=2 * len(list(feature.geometry.coords))) for feature in self.features}
        possible_duplicate_set = UnionFind() 
        duplicate_feat_set = UnionFind()
        for feature1 in self.features:
            boundary_polygon1 = buffer_polygon_map[feature1.id]
            for feature2_id in neighbouring_feature_map[feature1.id]:
                if feature1.id != feature2_id:
                    feature2 = self.feature_map[feature2_id]
                    boundary_polygon2 = buffer_polygon_map[feature2_id]
                    if boundary_polygon1.contains(feature2.geometry) and boundary_polygon2.contains(feature1.geometry):
                        duplicate_feat_set.union(duplicate_feat_set[feature1.id], duplicate_feat_set[feature2.id])
                        logger.debug('Found duplicate lines:{} and {}'.format(feature1.id, feature2_id))
                    else:
                        union_area = boundary_polygon1.union(boundary_polygon2).area.real
                        intersection_area = boundary_polygon1.intersection(boundary_polygon2).area.real
                        if union_area * self.threshold < intersection_area:
                            possible_duplicate_set.union(possible_duplicate_set[feature1.id], possible_duplicate_set[feature2.id])
                            logger.debug('Polygon buffers for lines:{} and {} overlap more than threshold'.
                                         format(feature1.id, feature2.id))
        
        # remove duplicates and save to file...
        duplicate_features_gs = self.save_duplicate_features(duplicate_feat_set)
        # save possible duplicates to file for manual inspection...
        possible_duplicate_gs = self.save_possible_duplicates(possible_duplicate_set)
        # save features which have no duplicates to a separate file...  
        remaining_features_gs = self.save_remaining_features(possible_duplicate_set, duplicate_feat_set)
        # save all features .. 
        all_features = duplicate_features_gs + possible_duplicate_gs + remaining_features_gs 
        for i, feature in enumerate(all_features):
            feature.id = str(i)
        gps_io_utils.write_gdf_json(GeoDataFrame(all_features), self.all_feature_filename)
            
    def process_data(self):
        self.merge_duplicate_lines()

    def supported_geometries(self):
        return set(['LineString'])

# if __name__ == '__main__':
#     line_merger = DuplicateLineDetector('/home/sanjeev/workspaces/shapes/aschheim/linestrings.geojson',
#                              '/home/sanjeev/workspaces/shapes/aschheim/test', category='road edge', threshold=0.3,
#                               boundary_length=2)
#     line_merger.process_data()

import os

import geopandas as gps
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from objectdetection.utils import logger, shape_utils
from objectdetection.utils.preprocessing_utils import get_metadata, get_source_feature_ids

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils.io import gps_io_utils


class DuplicatePolygonDetector(BasePreprocessor):
    '''
     Detects duplicate lines in a geojson and writes the detected features in three separate files.
     TODO: There is scope for refactoring here, some methods can be abstracted out between 
     DuplicatePolygonDetector and DuplicateLineDetector.
    '''
    
    
    def __init__(self, input_dir, out_folder_path, **kwargs):
        data_category = kwargs['category']
        threshold = kwargs['threshold']
        self.META_DATA = get_metadata(self.__class__.__name__, data_category)
        out_data_category = data_category.replace(' ', '_')
        input_file_path = os.path.join(input_dir, '{}-Polygon.geojson'.format(out_data_category))
        self.input_data = gps.read_file(input_file_path)
        self.data_category = data_category
        self.category_data = self.input_data
        if data_category is not None:
            self.category_data = self.category_data[self.input_data.category == data_category]
        self.category_data = self.category_data[self.category_data.geometry.type == 'Polygon']
        self.threshold = threshold
        self.features = [feature for _, feature in self.category_data.iterrows()]
        self.feature_map = {feature.id: feature for feature in self.features}
        # Out filenames...
        
        self.duplicate_feature_filename = os.path.join(out_folder_path, '{}-Polygon-duplicate.geojson'.
                                                format(out_data_category))
        self.possible_duplicate_filename = os.path.join(out_folder_path, '{}-Polygon-possible_duplicate.geojson'.
                                                format(out_data_category))
        self.other_feature_filename = os.path.join(out_folder_path, '{}-Polygon-other.geojson'.format(out_data_category))
        self.all_feature_filename = os.path.join(out_folder_path, '{}-Polygon.geojson'.format(out_data_category))
    
    def save_possible_duplicates(self, possible_dup_feature_map):
        possible_dup_features_gs = []
        i = 0 
        for _, feature_ids in possible_dup_feature_map.items():
            for feature_id in feature_ids:
                feature = self.feature_map[feature_id]
                source_feature_ids = get_source_feature_ids([feature])
                data = {'geometry' : feature.geometry, 'source_features' : ','.join(source_feature_ids),
                        'id' : str(i)}
                data.update(self.META_DATA)
                possible_dup_features_gs.append(GeoSeries(data))
                i += 1
        
        possible_dup_features_gdf = GeoDataFrame(possible_dup_features_gs)
        gps_io_utils.write_gdf_json(possible_dup_features_gdf, self.possible_duplicate_filename)
        return possible_dup_features_gs


    def save_remaining_features(self, possible_duplicate_feature_ids, duplicate_feature_ids, failed_features):
        all_feature_ids = set(map(lambda x : x.id, self.features))
        other_feature_ids = all_feature_ids.difference(duplicate_feature_ids.union(possible_duplicate_feature_ids))
        other_feature_ids = other_feature_ids.union(failed_features)
        logger.info('Total features:{}, detected duplicate features:{}, possible duplicate features:{}, other features:{}'.
                    format(len(self.features), len(duplicate_feature_ids), len(possible_duplicate_feature_ids), len(other_feature_ids)))
        other_features_gs = []
        for i, other_feature_id in enumerate(other_feature_ids):
            other_feature = self.feature_map[other_feature_id]
            source_feature_ids = get_source_feature_ids([other_feature])
            data = {'geometry' : other_feature.geometry, 'source_features' : ','.join(source_feature_ids),
                    'id' : str(i)}
            data.update(self.META_DATA)
            other_features_gs.append(GeoSeries(data))
        
        other_features_gdf = GeoDataFrame(other_features_gs)
        gps_io_utils.write_gdf_json(other_features_gdf, self.other_feature_filename)
        return other_features_gs

    def save_overlapping_features(self, intersecting_polygon_map, overlapping_fgeometries):
        overlapping_geo_series = []
        for i, item in enumerate(overlapping_fgeometries.items()):
            parent_id, fgeometry = item
            source_features = [self.feature_map[feature_id] for feature_id in intersecting_polygon_map[parent_id]]
            source_feature_ids = get_source_feature_ids(source_features)
            data = {'geometry' : fgeometry, 'source_features' : ','.join(source_feature_ids),
                    'id' : str(i)}
            data.update(self.META_DATA)
            geo_series = GeoSeries(data)
            overlapping_geo_series.append(geo_series)
        
        overlapping_features_gdf = gps.GeoDataFrame(overlapping_geo_series)
        gps_io_utils.write_gdf_json(overlapping_features_gdf, self.duplicate_feature_filename)
        return overlapping_geo_series

    '''
    Removes duplicate polygons which have common area greater than threshold. Currently all duplicate candidates are merged 
    into one by taking the union of all. Other approach could be to select the polygon with most points. 
    '''
    def merge_duplicate_polygons(self):
        overlapping_polygon_map, intersecting_polygon_map, _ = shape_utils.get_overlapping_polygons(self.features, self.threshold)
        overlapping_fgeometries, failed_features = shape_utils.get_merged_featgeo(self.features, overlapping_polygon_map)
        # save overlapping features...
        overlapping_gs = self.save_overlapping_features(overlapping_polygon_map, overlapping_fgeometries)
        # save possible duplicate features...
        possible_dup_features_gs = self.save_possible_duplicates(intersecting_polygon_map)
        # save other features....
        possible_duplicate_ids = {value for values in intersecting_polygon_map.values() for value in values}
        duplicate_feature_ids = {value for values in overlapping_polygon_map.values() for value in values}
        other_features_gs = self.save_remaining_features(possible_duplicate_ids, duplicate_feature_ids, failed_features)
        # save all features .. 
        all_features = overlapping_gs + possible_dup_features_gs + other_features_gs 
        for i, feature in enumerate(all_features):
            feature.id = str(i)
        gps_io_utils.write_gdf_json(GeoDataFrame(all_features), self.all_feature_filename)
    
    def process_data(self):
        self.merge_duplicate_polygons()
        
    def supported_geometries(self):
        return set(['Polygon'])
    
# if __name__ == '__main__':
#     duplicate_detector = DuplicatePolygonDetector('/home/sanjeev/workspaces/shapes/aschheim/guardrails/merged.geojson',
#                                                   '/home/sanjeev/workspaces/shapes/aschheim/guardrails/', category='guardrails',
#                                                    threshold=0.8)
#     duplicate_detector.process_data()

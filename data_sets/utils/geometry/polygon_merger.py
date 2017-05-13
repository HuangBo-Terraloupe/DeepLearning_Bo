import os

import geopandas as gps
from geopandas.geoseries import GeoSeries
from objectdetection.utils import geometry
from objectdetection.utils.preprocessing_utils import get_source_feature_ids, get_metadata

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils.io import gps_io_utils


class PolygonMerger(BasePreprocessor):
    '''
    Merges multiple polygons to one if the common area of the two polygons is above a threshold.
    '''
    
    def __init__(self, input_dir, outfolder, **kwargs):
        data_category = kwargs['category']
        self.bbuffer = 0.4
        self.META_DATA = get_metadata(self.__class__.__name__, data_category)
        out_data_category = data_category.replace(' ', '_')
        filename = '{}-Polygon.geojson'.format(out_data_category)
        input_file_path = os.path.join(input_dir, filename)
        self.input_data = gps.read_file(input_file_path)
        self.target_geojson_filename = os.path.join(outfolder, filename)
        self.data_category = data_category
        self.category_data = self.input_data
        if len(self.category_data) == 0:
            raise Exception('Empty input file')
        if data_category is not None:
            self.category_data = self.category_data[self.input_data.category == data_category]
        self.category_data = self.category_data[self.category_data.geometry.type == 'Polygon']
        self.features = [feature for _, feature in self.category_data.iterrows()]
        self.feature_map = {feature.id: feature for feature in self.features}
            
    def merge_polygons(self):
        '''
        Merges polygons intersecting polygons into one. This is achieved by following steps
        1. Find the intersecting polygons.
        2. Merge polygon geometries for overlapping polygon.
        3. Dump the geo json of new features to a file.
        '''
        intersecting_polygon_map = geometry.get_overlapping_polygons(self.features, 0, bbuffer=self.bbuffer)[0]
        merged_feature_set = {feature_id for feature_ids in intersecting_polygon_map.values() for feature_id in feature_ids}
        all_feature_set = {feature.id for feature in self.features}
        remaining_features = all_feature_set - merged_feature_set
        for feature_id in remaining_features:
            intersecting_polygon_map[feature_id] = [feature_id]
#             merged_fgeometries[feature_id] = self.feature_map[feature_id].geometry
        merged_fgeometries, failed_features = geometry.get_merged_featgeo(self.features, intersecting_polygon_map, self.bbuffer)
        for feature_id in failed_features:
            merged_fgeometries[feature_id] = self.feature_map[feature_id].geometry
            intersecting_polygon_map[feature_id] = [feature_id]
            
        out_geo_series = []
        count = 0
        for item in merged_fgeometries.items():
            parent_id, fgeometry = item
            if fgeometry.type == 'MultiPolygon':
                for geo in fgeometry.geoms:
                    data = {'geometry' : geo, 'source_features' : ','.join([parent_id]), 'id' : str(count)}
                    data.update(self.META_DATA)
                    geo_series = GeoSeries(data)
                    out_geo_series.append(geo_series)
                    count += 1
            else:
                source_features = [self.feature_map[feature_id] for feature_id in intersecting_polygon_map[parent_id]]
                source_feature_ids = get_source_feature_ids(source_features)
                data = {'geometry' : fgeometry, 'source_features' : ','.join(source_feature_ids), 'id' : str(count)}
                data.update(self.META_DATA)
                geo_series = GeoSeries(data)
                out_geo_series.append(geo_series)
                count += 1
                
        out_geo_series = [gs for gs in out_geo_series if not gs['geometry'].is_empty]
        geo_data = gps.GeoDataFrame(out_geo_series)
        # write merged polygons to geo json...
        gps_io_utils.write_gdf_json(geo_data, self.target_geojson_filename)
        
    def process_data(self):
        self.merge_polygons()
        
    def supported_geometries(self):
        return set(['Polygon'])

# if __name__ == '__main__':
#     polygon_merger = PolygonMerger('/home/sanjeev/workspaces/shapes/aschheim/duplicate/',
#                         '/home/sanjeev/workspaces/shapes/aschheim/merge/',
#                          category='vertical object')
#     polygon_merger.process_data()

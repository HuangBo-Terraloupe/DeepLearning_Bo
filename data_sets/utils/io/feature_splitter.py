import os

import geopandas as gps
from geopandas.geodataframe import GeoDataFrame
from objectdetection.utils import logger

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils.io import gps_io_utils

sentinel = object()


class CategoryAndGeometrySplitter(BasePreprocessor):
    '''
    Splits a file named input.geojson from indir directory by categories. 
    '''
    
    def __init__(self, indir, outdir, **kwargs):
        self.infile = os.path.join(indir, 'input.geojson') 
        self.outdir = outdir
        
    def process_data(self):
        input_data = gps.read_file(self.infile)
        all_features = [feature for _, feature in input_data.iterrows()]
        category_map = split_by_attr(all_features, 'category')
        
        logger.info('Total categories:{}'.format(len(category_map.keys())))
        for category_item in category_map.iteritems():
            geometry_type_map = split_by_attr(category_item[1], 'geometry.type')
            logger.info('Total geometries:{} for category:{}'.format(geometry_type_map.keys(), category_item[0]))
            for geometry_item in geometry_type_map.iteritems():
                category = category_item[0]
                geometry_type, features = geometry_item
                geo_dataframe = GeoDataFrame(features)
                out_filename = os.path.join(self.outdir, '{}-{}.geojson'.
                                            format(category.replace(' ', '_'), geometry_type))
                gps_io_utils.write_gdf_json(geo_dataframe, out_filename)

def split_by_attr(objects, split_property):
    '''
    Groups the obj by split_property attribute.
    '''
    obj_map = {}
    for obj in objects:
        property_value = rgetattr(obj, split_property, 'unmatched')
        property_value = 'unmatched' if property_value is None else property_value 
        if property_value not in obj_map:
            obj_map[property_value] = []
        obj_map[property_value].append(obj)
    return obj_map

def rgetattr(obj, attr, default=sentinel):
    '''
    Finds attribute values from nested objects.
    '''
    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return reduce(_getattr, [obj] + attr.split('.'))

# if __name__ == '__main__':
#     feature_splitter = CategoryAndGeometrySplitter('/home/sanjeev/workspaces/shapes/aschheim/gt_labels.geojson',
#                                        '/home/sanjeev/workspaces/shapes/aschheim/split')
#     feature_splitter.process_data()

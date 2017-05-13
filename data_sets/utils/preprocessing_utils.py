import getpass
import os

import datetime
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from objectdetection.utils import logger

from objectdetection.driver.base import OutFileInfo
from objectdetection.utils.io import gps_io_utils


def get_source_feature_ids(features):
    '''
    Return the list of source feature ids for all features.
    '''
    src_feature_ids = []
    for feature in features:
        if hasattr(feature, 'source_features'):
            src_feature_ids.extend(feature.source_features.split(','))
        else:
            src_feature_ids.append(feature.id)
    return src_feature_ids

def get_metadata(algorithm_name, data_category):
    '''
    Return the meta data properties which are to added to each feature after running the 
    algorithm.
    '''
    meta_data = {'creator' : getpass.getuser(), 'algorithm' : algorithm_name,
                          'derived' : 'true', 'created_at' : str(datetime.datetime.now()),
                          'category':data_category}
    return meta_data

def get_all_files(input_dir, suffix='.geojson'):
    all_files = os.listdir(input_dir)
    file_infos = []
    for filename in all_files:
        filename_abs = os.path.join(input_dir, filename)
        if filename.endswith(suffix) and os.path.isfile(filename_abs):
            split = filename[:-len(suffix)].split('-')
            category, geometry_type = None, None
            if len(split) == 2:
                category, geometry_type = split[:2]
            elif len(split) > 2:
                logger.info('Skipping file:{} as it is intermediate file'.format(filename))
                continue
            file_infos.append(OutFileInfo(filename_abs, category, geometry_type))
            
    return file_infos

def save_features_list(meta_data, feature_map, new_features_list, out_filename):
    '''
    Saves list of tuples of form (feature_id, geometry) with index as new id, source id from 
    old feature id sources and appends meta data to each feature. The output file is written to out_filename.
    '''
    normalized_features_gs = []
    for i, item in enumerate(new_features_list):
        feature_id, polygon = item
        source_feature_ids = get_source_feature_ids([feature_map[feature_id]])
        data = {'geometry' : polygon, 'source_features' : ','.join(source_feature_ids), 'id' : str(i)}
        data.update(meta_data)
        normalized_features_gs.append(GeoSeries(data))
    normalized_gdf = GeoDataFrame(normalized_features_gs)
    gps_io_utils.write_gdf_json(normalized_gdf, out_filename)
    

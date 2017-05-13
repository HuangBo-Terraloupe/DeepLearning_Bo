'''
All utility methods related to geo pandas io.
'''

import json

import geopandas as gps

from objectdetection.utils import logger


def write_gdf_json(geo_dataframe, out_filename, should_format=True):
    '''
    Writes GeoDataFrame object as geo json to @param out_filename. Additionally formats 
    the json unless @param should_format is False.
    '''
    geo_json = geo_dataframe.to_json()
    if should_format:
        parsed_geo_json = json.loads(geo_json) 
        geo_json = json.dumps(parsed_geo_json, indent=4, sort_keys=True)
    with open(out_filename, 'w') as fw:
        fw.write(geo_json)
    logger.info('Successfully wrote geojson to file:{}'.format(out_filename))

def shp_to_geojson(shp_filename, geojson_filename, should_format=True):
    '''
    Converts .shp file to .geojson.
    '''
    input_data = gps.read_file(shp_filename)
    write_gdf_json(input_data, geojson_filename, should_format)
    

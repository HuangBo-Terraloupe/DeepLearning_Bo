import os

import geopandas as gps
from geopandas.geodataframe import GeoDataFrame
from objectdetection.utils.preprocessing_utils import get_all_files

from objectdetection.driver.base import BasePreprocessor
from objectdetection.utils.io import gps_io_utils


class FeatureMerger(BasePreprocessor):
    '''
    Merges all gejson files from input dir and writes them to output.geojson in outdir.
    '''
    
    def __init__(self, indir, outdir, **kwargs):
        self.indir = indir
        self.outdir = outdir
        
    def process_data(self):
        all_files = get_all_files(self.indir)
        all_features = []
        for file_info in all_files:
            input_data = gps.read_file(file_info.filename)
            features = [feature for _, feature in input_data.iterrows()]
            all_features.extend(features)
        out_filename = os.path.join(self.outdir, 'output.geojson')
        gps_io_utils.write_gdf_json(GeoDataFrame(all_features), out_filename)

# if __name__ == '__main__':
#     feature_merger = FeatureMerger('test', '/home/sanjeev/workspaces/shapes/aschheim/input', merge_dir='/home/sanjeev/workspaces/shapes/aschheim/merge/')
#     feature_merger.process_data()

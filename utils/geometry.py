import os
import json
import rasterio
import numpy as np
import geopandas as gpd

from rasterio.features import shapes
from shapely.geometry import polygon
from shapely.geometry import shape, box, Point
from geopandas import GeoSeries, GeoDataFrame


def write_gdf_json(geo_df, out_filename, epsg_out=32632, should_format=False):
    """ Function to save (geo)data as *.geojson (needs to have .crs information!)
    Args:
        geo_df: GeoDataFrame in memory
        out_filename: full output path for *.geojson file
        epsg_out: If EPSG is not mentioned in the input GeoDataFrame, then this EPSG will be used.
        should_format: boolean that allows for readable indentation in the output json
    Returns:
        *.geojson file in output directory based on EPSG of vector data

    """

    # build crs header information to the header (easier I/O)
    if geo_df.crs is not None:
        epsg_out = int(geo_df.crs['init'].replace('epsg:', ''))

    header = {
        'type': 'name',
        'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % epsg_out}
    }

    # add header to dictionary
    result = json.loads(geo_df.to_json())
    result['crs'] = header

    # indent based on formatting decision
    if should_format:
        indent = 4
    else:
        indent = None

    # save as *.geojson
    with open(out_filename, 'w') as fp:
        json.dump(result, fp, indent=indent, sort_keys=True)

def explode(gdf):
    """
    Explodes a geodataframe

    Will explode muti-part geometries into single geometries. Original index is
    stored in column level_0 and zero-based count of geometries per multi-
    geometry is stored in level_1

    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries

    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with a new index
                                 and two new columns: level_0 and level_1

    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    #gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf2.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out

def combine_multi_rasters(raster_1, raster_2, output_raster):
    im1 = rasterio.open(raster_1)
    im1_ndarray = im1.read(1).astype(np.bool)

    im2 = rasterio.open(raster_2)
    im2_ndarray = im2.read(1).astype(np.bool)

    im_ndarray = im1_ndarray + im2_ndarray
    # im_ndarray[im_ndarray > 0] = 1
    # im_ndarray.astype(np.uint8)
    #
    # print(im_ndarray.max(), im_ndarray.min())
    with rasterio.open(output_raster, 'w', **im1.profile) as ff:
        ff.write()


if __name__ == '__main__':
    raster_1 = '/home/bo_huang/rasters/segmentation_results_hexagon_roads_pitsburgh.tif'
    raster_2 = '/home/bo_huang/rasters/segmentation_results_hexagon_roads_v2_pitsburgh.tif'
    output_raster = '/home/bo_huang/rasters/combine.tif'
    combine_multi_rasters(raster_1, raster_2, output_raster)
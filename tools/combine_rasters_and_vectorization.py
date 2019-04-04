import os
import json
import rasterio
import multiprocessing
import numpy as np

from glob import glob
from rasterio.features import shapes
from shapely.geometry import shape, box
from geopandas import GeoSeries, GeoDataFrame


def read_category_input(category_input):
    """
    Function for simplifying the input of category names of classification and desired bounding box visualizations

    Args:
        category_input: file with comma delimited list of <pixel_value>, <category_name>, <bbox flag>, <kernel width>, <min_area>

    Returns:
        category_info: dictionary containing formatted information by category

    """

    # open file -> get category names and bounding box categories for visualization
    with open(category_input, 'r') as f:
        raw = f.readlines()

    # get header parameters
    headers = [r.strip() for r in raw[0].strip().split(',')]

    # organize information into a dictionary
    category_info = []
    for r in raw[1:]:
        temp_dict = dict(zip(headers, [x.strip() for x in r.strip().split(',')]))
        temp_dict['pixel_value'] = int(temp_dict['pixel_value'])

        if 'vectorize' in temp_dict:
            temp_dict['vectorize'] = bool(int(temp_dict['vectorize']))
        else:
            temp_dict['vectorize'] = True

        if 'bbox_flag' in temp_dict:
            temp_dict['bbox_flag'] = bool(int(temp_dict['bbox_flag']))
        else:
            temp_dict['bbox_flag'] = None

        if 'kernel_size' in temp_dict:
            temp_dict['kernel_size'] = int(temp_dict['kernel_size'])
        else:
            temp_dict['kernel_size'] = None

        if 'min_area' in temp_dict:
            temp_dict['min_area'] = float(temp_dict['min_area'])
        else:
            temp_dict['min_area'] = None

        category_info.append(temp_dict)

    return category_info

def write_json(data, output_name, output_dir, geo_flag=True, indent=4):
    """
    Function to save (geo)data as *.geojson (needs to have .crs information!)

    Args:
        data: GeoDataFrame in memory
        output_name: desired output name
        output_dir: desired output folder for saving final output data
        geo_flag: boolean for saving *.geojson + CRS info
        indent: desired indent level for the output json

    Returns:
        *.geojson file in output directory based on EPSG of vector data

    """

    # if *.geojson is desired
    if geo_flag:

        # build crs header information to the header (easier I/O)
        epsg_out = int(data.crs['init'].replace('epsg:', ''))
        header = {
            'type': 'name',
            'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % epsg_out}
        }

        # add header to dictionary
        result = json.loads(data.to_json())
        result['crs'] = header

        # build file name
        file_ext = 'geojson'

    else:
        result = data

        # build file name
        file_ext = 'json'

    # save as *.geojson
    with open(os.path.join(output_dir, '%s.%s' % (output_name, file_ext)), 'w') as fp:
        json.dump(result, fp, indent=indent, sort_keys=True)


def vectorize_image(input_dir, output_dir, category, threshold):
    """
    Core function for converting raster to vector features in a *.pickle file

    Args:
        input_params: zipped list containing -> (fid, temp_dir, output_dir, category_info)

    Returns:
        *.pickle file containing GeoDataFrame containing shapes by category

    """

    # unpack inputs
    raster_list = glob(input_dir + '/*.tif')
    for id, fid in enumerate(raster_list):

        if id == 0:
            # load merged classified raster -> vectorize
            im = rasterio.open(fid)
            data = im.read(1)
        else:
            # load merged classified raster -> vectorize
            im = rasterio.open(fid)
            temp = im.read(1)
            data = data + temp

    data[data > threshold] = 1
    data[data <= threshold] = 0
    mask = np.array(data, dtype=np.bool)

    # contour raster image -> build polygons
    temp = shapes(data, mask, transform=im.transform)


    # compile results together as shapely geometry -> build GeoSeries
    out = []
    for t, v in temp:

        v = int(v)
        geo = shape(t)
        geo = box(*geo.bounds)

        out.append(GeoSeries({'geometry': geo,
                              'num': v,
                              'category': category}))

    # only write out features if they exist!
    if len(out) > 0:
        out = GeoDataFrame(out).sort_values(by='num', ascending=True)
        out.crs = im.crs  # get epsg from input file

        # write file to *.geojson
        out.to_file(driver='ESRI Shapefile', filename=os.path.join(output_dir, 'combined.shp'))

if __name__ == '__main__':
    input_dir = '/home/bo_huang/rasters'
    output_dir = '/home/bo_huang/rasters'
    category_info = 'lane_markings'
    threshold = '0'
    vectorize_image(input_dir, output_dir, category_info, threshold)
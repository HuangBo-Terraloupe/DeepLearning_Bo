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


def multiprocessing_map(fn, params, n_cores=1):
    """
    Helper function for mapping functions based on number of cores (i.e. with or without multiprocessing call)

    Args:
        fn: input function
        params: zipped list of input arguments
        n_cores: number of cpu cores for process

    Returns:
        function call based on input/params

    """

    map_fn = map
    if n_cores > 1:
        map_fn = multiprocessing.Pool(n_cores).map

    return list(map_fn(fn, params))

def process_vectorization(input_dir, output_dir, category_info, cores=4):

    # initialize jobs
    jobs = []

    # get all merged images output directory
    fids = glob(input_dir + '/*.tif')
    fids.sort()
    print(fids)

    # set up params via zip
    n = len(fids)
    file_for_each_worker = int(n / cores)

    # category_info
    category_info = read_category_input(category_info)

    for i in range(cores):
        if i != cores -1:
            p = multiprocessing.Process(
                target=vectorize_image,
                args=(fids[i*file_for_each_worker:(i+1)*file_for_each_worker], output_dir, category_info,))
        else:
            p = multiprocessing.Process(
                target=vectorize_image,
                args=(fids[i*file_for_each_worker:], output_dir, category_info,))

        jobs.append(p)
        p.start()


def vectorize_image(raster_list, output_dir, category_info):
    """
    Core function for converting raster to vector features in a *.pickle file

    Args:
        input_params: zipped list containing -> (fid, temp_dir, output_dir, category_info)

    Returns:
        *.pickle file containing GeoDataFrame containing shapes by category

    """

    # unpack inputs
    for id, fid in enumerate(raster_list):
        print(id)

        # load merged classified raster -> vectorize
        im = rasterio.open(fid)
        data = im.read(1)

        # build "Truth" mask for everything that should be included in the vectorization
        mask = np.zeros(data.shape, dtype=np.bool)
        vec_vals = [c['pixel_value'] for c in category_info if c['vectorize']]
        for vec_val in vec_vals:
            mask[data == vec_val] = 1
        mask = np.array(mask, dtype=np.bool)

        # contour raster image -> build polygons
        temp = shapes(data, mask, transform=im.transform)

        # check if any objects should be visualized as bounding boxes
        bb_flag = any([c['bbox_flag'] for c in category_info])
        if bb_flag:
            bb_vals = [c['pixel_value'] for c in category_info if c['bbox_flag']]

        # compile results together as shapely geometry -> build GeoSeries
        out = []
        for t, v in temp:

            v = int(v)
            geo = shape(t)
            if bb_flag and any([v == b for b in bb_vals]):
                geo = box(*geo.bounds)

            out.append(GeoSeries({'geometry': geo,
                                  'num': v,
                                  'category': [c['category_name'] for c in category_info if c['pixel_value'] == v][0]}))

        # only write out features if they exist!
        if len(out) > 0:
            out = GeoDataFrame(out).sort_values(by='num', ascending=True)
            out.crs = im.crs  # get epsg from input file

            # write file to *.geojson
            pather, name = os.path.split(fid)
            write_json(out, name.split('.')[0], output_dir, geo_flag=True, indent=None)

if __name__ == '__main__':
    input_dir = '/home/bo_huang/rasters/pools'
    output_dir = '/home/bo_huang/rasters/pools'
    category_info = '/home/bo_huang/rasters/pools/pools_category'
    process_vectorization(input_dir, output_dir, category_info, cores=1)
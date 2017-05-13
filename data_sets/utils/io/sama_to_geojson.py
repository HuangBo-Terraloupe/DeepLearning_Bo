import json
import os

import click
import rasterio
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon
from objectdetection.utils import logger
from objectdetection.utils.image.img_tiling import parse_img_id

from objectdetection.utils.io import gps_io_utils


def get_affine_transform(base_tiff_dir, tiff_id, img_ext):
    file_path = os.path.join(base_tiff_dir, tiff_id + '.%s' % img_ext)
    with rasterio.open(file_path) as tiff_file:
        affine = tiff_file.affine
    return affine


def get_image_shape(base_tiff_dir, tiff_id, img_ext):
    file_path = os.path.join(base_tiff_dir, tiff_id + '.%s' % img_ext)
    with rasterio.open(file_path) as tiff_file:
        width = tiff_file.width
        height = tiff_file.height
    return (height, width)

def convert_tiles_to_wc(tiff_dir, points, image_filename, imagetype, block_shape):
    tiff_id, row, column = parse_img_id(image_filename)
    row *= block_shape[0]
    column *= block_shape[1]
    affine = get_affine_transform(tiff_dir, tiff_id, imagetype)
    new_coords = [affine * (column + coord[0] + 0.5, row + coord[1] + 0.5) for coord in points]
    return new_coords


def convert_tiles_to_image_coordinates(points, image_filename, block_shape):
    """ Convert annotation coordinates in image tile coordinate system to image
    coordinate system.
    """
    tiff_id, row, column = parse_img_id(image_filename)
    row *= block_shape[0]
    column *= block_shape[1]
    new_coords = [(column + coord[0] + 0.5, row + coord[1] + 0.5) for coord in points]
    return new_coords


def convert_to_wc(image_dir, points, image_filename, imagetype):
    filename = os.path.split(image_filename)[-1]
    img_id = os.path.splitext(filename)[0]
    affine = get_affine_transform(image_dir, img_id, imagetype)
    new_coords = [affine * (coord[0] + 0.5, coord[1] + 0.5) for coord in points]
    return new_coords

@click.command()
@click.argument("labeled_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--tiff_dir", default=None, type=click.Path())
@click.option("--namefield", default="name")
@click.option("--imagetype", default="tif")
@click.option("--block-shape", type=(int, int), default=(0, 0))
def convert_to_geojson(labeled_file, out_file, tiff_dir, namefield, imagetype, block_shape):
    '''
    Converts labeled json file provided by sama to geojson file.
    @param tiff_dir: Directory which contains the tiff files for the labeled data.
    @param labeled_file: Annotated json file with labels.
    @param out_file: Name of output geojson file.
    '''
    gs_features = []
    with open(labeled_file) as f:
        parsed_json = json.load(f)
        tasks = parsed_json['tasks']
        logger.info('Total tasks:{}'.format(len(tasks)))
        unique_id = 0
        for task in tasks:
            if 'tags' in task['Tags']:
                # This was the original format as observed in the roof project.
                tags = task['Tags']['tags']
            elif 'Tag' in task['Tags']:
                # There seems to be different formats coming from Samasource. This is
                # the format we observed in the Autobahn project.
                tags = task['Tags']['Tag']

            task_data = dict(task)
            task_data.pop('Tags')
            #logger.debug('Processing tags for image_id:{}'.format(task_data['name']))
            for tag in tags:
                shape_type = tag['tags']['type']
                if block_shape[0] > 0:
                    if tiff_dir is not None:
                        wc_points = convert_tiles_to_wc(tiff_dir, tag['tags']['points'],
                                                        task[namefield],
                                                        imagetype,
                                                        block_shape)
                    else:
                        points = tag['tags']['points']
                        wc_points = convert_tiles_to_image_coordinates(points,
                                                                       task[namefield],
                                                                       block_shape)
                else:
                    wc_points = convert_to_wc(tiff_dir, tag['tags']['points'],
                                              task[namefield], imagetype)

                if shape_type == 'line':
                    geometry = LineString(wc_points)
                elif shape_type == 'polygon':
                    geometry = Polygon(wc_points)
                elif shape_type == 'rect':
                    geometry = Polygon(wc_points).convex_hull
                else:
                    logger.warn('Unknown category:{} for image_id:{}'.format(shape_type, task_data['name']))
                    continue

                category = tag['tags']['tag']
                feature_data = {'geometry':geometry, 'category': category, 'tid': unique_id}
                # tid - unique id of each shape feature such that it does not clash with internal id attribute of
                # geopandas.
                feature_data.update(task_data)
                gs = GeoSeries(feature_data)
                gs_features.append(gs)
                unique_id += 1

        gps_io_utils.write_gdf_json(GeoDataFrame(gs_features), out_file)

if __name__ == '__main__':
    convert_to_geojson()
#     tiff_dir = '/home/sanjeev/Downloads/datasets/regensburg/tiff_data'
#     labeled_file = '/home/sanjeev/Downloads/datasets/regensburg/labeled_data_sama.json'
#     out_file = '/home/sanjeev/Downloads/datasets/regensburg/labeled_data.geojson'
#     convert_to_geojson(tiff_dir, labeled_file, out_file)

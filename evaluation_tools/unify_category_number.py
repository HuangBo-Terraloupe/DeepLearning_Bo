import geopandas as gpd
import json
import os


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
    with open(os.path.join(output_dir, '%s.%s' % (output_name, file_ext)), 'wb') as fp:
        json.dump(result, fp, indent=indent, sort_keys=True)

def unify_category_num(geojson_file, class_mapping, output_name, output_dir):
    data = gpd.read_file(geojson_file)
    for index, row in data.iterrows():
        data['category_num'][index] = class_mapping[row['category']]
    write_json(data, output_name, output_dir)


if __name__ == '__main__':
    geojson_file = '/home/huangbo/evaluation/prediction/prediction.geojson'
    output_name = 'pre_unify'
    output_dir = '/home/huangbo/evaluation/prediction'
    class_mapping = {'UP_TO_5':0,
                     'UP_TO_10':1,
                     'UP_TO_25':2,
                     'UP_TO_75':3,
                     'ABOVE_75':4,
                     'bg':5
                     }
    unify_category_num(geojson_file, class_mapping, output_name, output_dir)
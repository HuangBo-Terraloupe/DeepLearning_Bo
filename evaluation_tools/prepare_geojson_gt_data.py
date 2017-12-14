import os
import json
import yaml

from shapely.geometry import Polygon
from geopandas import GeoSeries, GeoDataFrame

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

def create_geojson(yml_file, class_mapping, output_name, output_dir):
    output = []
    id = 0
    with open(yml_file, 'r') as fp:
        spec = yaml.load(fp.read())
    test_labels = spec['testing']['labels']
    if len(test_labels) == 0:
        test_labels = spec['validation']['labels']
    print 'the number of evluation images is:', len(test_labels)

    for label in test_labels:
        data = json.loads(open(spec['prefix'] + label, "r").read())
        img_name = data['img_name']
        if len(data['bboxes']) ==0:
            output.append(GeoSeries({'id': id,
                                     'category': 'bg',
                                     'category_num':class_mapping[str(category)],
                                     'geometry': None,
                                     'img_name': img_name,
                                     }))
            id += 1
        else:
            for bbox in data['bboxes']:
                category = bbox['category']
                temp_poly = [[bbox['x1'], bbox['y1']],
                             [bbox['x1'], bbox['y2']],
                             [bbox['x2'], bbox['y2']],
                             [bbox['x2'], bbox['y1']]]

                output.append(GeoSeries({'id': id,
                                         'category': category,
                                         'category_num': class_mapping[str(category)],
                                         'geometry': Polygon(temp_poly),
                                         'img_name': img_name,
                                         }))


                id += 1

    df = GeoDataFrame(output)
    df.crs = {'init': 'epsg:32632'}
    write_json(df,
               output_name=output_name,
               output_dir=output_dir,
               geo_flag=True,
               indent=None
               )

if __name__ == '__main__':
    # class_mapping = {'UP_TO_5': 0,
    #                  'UP_TO_10': 1,
    #                  'UP_TO_25': 2,
    #                  'UP_TO_75': 3,
    #                  'ABOVE_75': 4,
    #                  'bg': 5
    #                  }
    # yml_file = '/home/huangbo/harvey/Building_damage_nofilter/building_damage.yml'
    # output_name = 'groundtruth'
    # output_dir = '/home/huangbo/harvey/evaluation/gt'
    class_mapping = {'tank': 0,
                     'bg': 1}
    yml_file = '/home/huangbo/tank_detection/dataset/tank.yml'
    output_name = 'groundtruth'
    output_dir = '/home/huangbo/tank_detection/evaluation/retina_net/gt'
    create_geojson(yml_file, class_mapping, output_name, output_dir)
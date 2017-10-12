import json
import numpy as np
import pandas as pd
from glob import glob
from shapely.geometry import Polygon


def get_data_from_json(json_folder, save_folder, filter_category):
    '''Create json file for every image patches

    Args:
        json_folder: path to the folder of json files, which contains all image annotations
        save_folder: the folder to save the json file for every image samples
        filter_category: a list of category that you want

    Returns:
        yml file

    '''

    merge_class = {'Rooftop Area | NOT Clear' : 'Rooftop Area',
                   'Rooftop Area | Clear' : 'Rooftop Area',
                   'Object | Solar Thermal Panel' : 'Panel',
                   'Object | Photovoltaic Panel' : 'Panel',
                   'Object | Chimney/Ventilation Pipe' : 'Chimney/Ventilation Pipe',
                   'Object | Roof Window' : 'Roof Window'
                   }

    json_files = glob(json_folder + "*.json")
    for json_file in json_files:
        data = json.load(open(json_file, "r"))
        df = pd.DataFrame(data["tasks"])

        for _, feature in df.iterrows():
            img_name = feature['name']
            annotation_data = {"img_name": img_name,
                               'bboxes': []
                               }
            for tag in feature["Tags"]["tags"]:
                category = tag["tags"]["tag"]
                if category not in filter_category:
                    continue
                points = tag["tags"]['points']
                try:
                    polygon = Polygon(np.asarray(points))
                except:
                    continue
                bbox = np.array(polygon.bounds, dtype=int)
                annotation_data['bboxes'].append(
                    {'category': merge_class[category],
                     'x1': bbox[0],
                     'x2': bbox[2],
                     'y1': bbox[1],
                     'y2': bbox[3],
                     }
                )
            print(len(annotation_data['bboxes']))
            # if len(annotation_data['bboxes']) == 0:
            #     continue
            # else:
            with open(save_folder + feature['name'][0:-3] + 'json', 'w') as fp:
                json.dump(annotation_data, fp)


if __name__ == '__main__':
    json_folder = '/home/huangbo/HuangBo_Projects/regensburg/json/'
    save_folder = '/home/huangbo/HuangBo_Projects/regensburg/json_background/'
    filter_category = ['Rooftop Area | NOT Clear', 'Rooftop Area | Clear', 'Object | Solar Thermal Panel',
                       'Object | Photovoltaic Panel', 'Object | Chimney/Ventilation Pipe', 'Object | Roof Window']
    get_data_from_json(json_folder, save_folder, filter_category)
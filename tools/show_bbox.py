import os
import cv2
import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

json_dir = '/home/huangbo/HuangBo_Projects/regensburg/test_json/'
json_files = glob(json_dir+"*.json")
img_dir = '/home/huangbo/HuangBo_Projects/regensburg/ortho/'

classes = { u'Rooftop Area | Clear': 1,
            u'Rooftop Area | NOT Clear': 1,
            u'Object | Solar Thermal Panel': 2,
            u'Object | Photovoltaic Panel': 2,
            u'Object | Chimney/Ventilation Pipe': 3,
            u'Object | Roof Window': 4
            }


def get_objects(feature, h, w):
    objects = []
    for tag in feature["Tags"]["tags"]:
        category = tag["tags"]["tag"]
        if category == "Roofridge Line":
            continue
        points = tag["tags"]["points"]
        if category == u'Object | Chimney/Ventilation Pipe' or category == u'Object | Satellite Dish':
            idx = ConvexHull(points).vertices
            points = np.array(points)[idx]
        area = Polygon(np.asarray(points)).area
        obj = [category, points, area]
        objects.append(obj)

    objects = sorted(objects, key=lambda l: l[2], reverse=True)

    final_objects = []
    for i, obj in enumerate(objects):
        category = obj[0]
        points = obj[1]
        if category in classes:

            label = classes[category]
            polygon = Polygon(np.asarray(points))

            # Remove vegetation over roof
            if label == 1:
                for o in enumerate(objects):
                    if o[0] == u'Object | Vegetation OVER roof':
                        polygon = Polygon(np.asarray(points)).difference(Polygon(np.asarray(o[1])))

                        # pts = polygon.exterior.coords[:]
            final_objects.append([label, polygon])

    n_instances = len(final_objects)
    gt_boxes = np.zeros((n_instances, 5))
    gt_masks = np.zeros((n_instances, h, w))
    complete_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(final_objects)):
        obj = final_objects[i]
        polygon = obj[1]

        bbox = polygon.bounds + (obj[0],)
        bbox = np.array(bbox)
        gt_boxes[i] = bbox

        points = polygon.exterior.coords[:]
        contour = np.array(points).reshape((-1, 2)).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 1)
        cv2.drawContours(complete_mask, [contour], 0, obj[0])
        gt_masks[i] = mask

    return n_instances, gt_boxes, gt_masks, complete_mask


def idx_to_name(idx):
    if idx == 1:
        return 'Rooftop_Area'
    if idx == 2:
        return 'Roof_Window'
    if idx == 3:
        return 'Solar_Thermal_Panel'
    if idx == 4:
        return 'Chimney/Ventilation_Pipe'
    else:
        return None




if __name__ == '__main__':
    save_dir = '/home/huangbo/HuangBo_Projects/regensburg/evaluation/gt/'

    data = json.load(open(json_files[0], "rb"))
    df = pandas.DataFrame(data["tasks"])
    idx = 0

    class_to_color = {'Rooftop_Area': np.array([150, 43, 30]),
                      'Roof_Window': np.array([177, 5, 207]),
                      'Solar_Thermal_Panel': np.array([221, 175, 35]),
                      'Chimney/Ventilation_Pipe': np.array([142, 213, 107]),
                      'bg': np.array([152, 242, 60])
                      }

    for _, feature in df.iterrows():

        name = feature["name"]
        image_path = os.path.join(img_dir, name)
        save_path = os.path.join(save_dir, name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        h, w = img.shape[0:2]
        n_instances, gt_boxes, gt_masks, complete_mask = get_objects(feature, h, w)
        print idx
        idx += 1


        if n_instances>=1:
            for i in range(n_instances):
                key = idx_to_name(int(gt_boxes[i][4]))
                cv2.rectangle(img,
                              (int(gt_boxes[i][0]), int(gt_boxes[i][1])),
                              (int(gt_boxes[i][2]), int(gt_boxes[i][3])),
                              (int(class_to_color[key][0]),
                               int(class_to_color[key][1]),
                               int(class_to_color[key][2])),
                              2
                              )


            cv2.imwrite(save_path, img)

            # plt.show()
            # plt.close()




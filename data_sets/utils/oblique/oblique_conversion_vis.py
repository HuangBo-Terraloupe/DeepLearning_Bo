import geopandas as gps
import numpy as np
import objectdetection.oblique.coordinate_conversion as cord
import rasterio
import skimage.io as sio
from affine import Affine
from scipy.spatial.distance import euclidean
from skimage.draw import circle, ellipse_perimeter, ellipse

from objectdetection.oblique.ori_parser import parse_ori

footprints = gps.read_file(
    "/Users/gerke/datasets/regensburg/05_Footprints/Nordblick.shp")
annotations = gps.read_file(
    "/Users/gerke/datasets/regensburg_oblique/regensburg_oblique_annotations.json")

roof_height = 6.0

for img_id in annotations.image_id.unique():
    ann = annotations[annotations.image_id == img_id]
    img = sio.imread("/Users/gerke/datasets/regensburg/jpeg/north/%s.jpg" %
                     img_id)
    K, R, C = parse_ori("/Users/gerke/data/regensburg/ori/%s.ori" % img_id)
    print C
    print "Image dimensions", img.shape

    footprint = footprints[footprints.BILD_NAME == img_id]
    footp = footprint.geometry.iloc[0]
    height = footprint.COG_Z.iloc[0]
    print height

    for i in range(ann.geometry.shape[0]):
        print i
        polygon = ann.geometry.iloc[i]
        x, y = np.array(polygon.centroid)
        # get lower point of bounding box
        y = polygon.bounds[1]

        x += 1635
        y += 1226

        # x = 8175
        # y = 6131
        # x = 0
        # y = 0

        print "Position of sat dish:", x, y

        rr, cc = circle(y, x, 10)
        img[rr, cc, :2] = 255

        # imshow(img[y:y+400, x:x+400])
        # imshow(img[y - 500:y, x - 500:x])
        obliq_vis = (img[y - 300:y + 300, x - 300:x + 300])

        sio.imsave("%s_%d_oblique.jpg" % (img_id, i), obliq_vis)

        height = cord.estimate_height([x, y], footprint, K, R, C)

        print "refined height", height

        # height = 330

        world_point = cord.img2world([x, y], K, R, C)
        world_point = cord.project_ground(world_point, C, height + roof_height)
        print "Estimated position:", world_point[0], world_point[1]

        print "Footprint coordinates:"
        for r in np.array(footp.exterior.coords)[:-1]:
            print r[0], r[1], r[2]
            print "Distance", euclidean(world_point[:2], r[:2])

        win_size = 200

        with rasterio.drivers():
            with rasterio.open(
                    "/Users/gerke/datasets/regensburg/jpeg/ortho.vrt") as ds:
                transform = ds.get_transform()
                af_transform = ~Affine.from_gdal(*transform)

                center_point = af_transform * world_point[:2]
                center_point = [int(center_point[0]), int(center_point[1])]

                window = (
                    (center_point[1] - win_size, center_point[1] + win_size),
                    (center_point[0] - win_size, center_point[0] + win_size))
                print window
                ortho_img = ds.read(window=window)
                ortho_img = np.rollaxis(ortho_img, 0, 3)
                print ortho_img.shape

        rr, cc = circle(win_size, win_size, 4)
        ortho_img[rr, cc, :2] = 255
        rr, cc = ellipse(win_size, win_size, yradius=70, xradius=20)
        ortho_img[rr, cc, 2] /= 80
        rr, cc = ellipse_perimeter(win_size, win_size, yradius=70, xradius=20)
        ortho_img[rr, cc, :2] = 255
        sio.imsave("%s_%d_ortho.jpg" % (img_id, i), ortho_img)

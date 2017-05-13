import geopandas as gps
import numpy as np
import objectdetection.oblique.coordinate_conversion as cord
import skimage.io as sio
from fiona.crs import from_epsg
from shapely.affinity import scale, translate
from shapely.geometry import Polygon
from skimage.draw import ellipse_perimeter

from objectdetection.oblique.ori_parser import parse_ori

source_crs = from_epsg(31468)
target_crs = from_epsg(4326)


footprints = gps.read_file(
    "/Users/gerke/datasets/regensburg/05_Footprints/Nordblick.shp")
annotations = gps.read_file(
    "/Users/gerke/data/regensburg_oblique/regensburg_sat_validation_yopeso_converted.geojson")

annotations['category'] = 'sat-dish'
roof_height = 6.0

rr, cc = ellipse_perimeter(0, 0, yradius=70, xradius=30)
xy = np.vstack([cc, rr]).T
mypoly = Polygon(xy)
mypoly = mypoly.buffer(1.0).buffer(-1.0)
simpl = mypoly.simplify(tolerance=1.5)
simpl = scale(simpl, 0.1, 0.1)


sat_dishes = []
for img_id in annotations.image_id.unique():
    print "Image:", img_id
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
        polygon = ann.geometry.iloc[i]
        x, y = np.array(polygon.centroid)
        # get lower point of bounding box
        y = polygon.bounds[1]

        x += 1635
        y += 1226

        # print "Position of sat dish:", x, y

        height = cord.estimate_height([x, y], footprint, K, R, C)
        # print "refined height", height

        world_point = cord.img2world([x, y], K, R, C)
        world_point = cord.project_ground(world_point, C, height + roof_height)
        print "Estimated position:", world_point[0], world_point[1]

        ellipse = translate(simpl, xoff=world_point[0], yoff=world_point[1])
        sat_dishes.append(ellipse)

result = gps.GeoDataFrame({'category': 'sat-dish', 'geometry': sat_dishes})
# result['geometry'] = gps.GeoSeries(sat_dishes, crs=source_crs)
result.crs = source_crs
# result = result.set_geometry('geometry')
with open("/Users/gerke/data/regensburg_oblique/sat_dishes.geojson", 'w') as fp:
    fp.write(result.to_crs(target_crs).to_json())

import os
import os.path

import click
import geopandas as gps
import numpy as np
import shapely.geometry as sg
import skimage.exposure
import skimage.io as sio
from skimage.draw import polygon_perimeter

from objectdetection.oblique.scene import Scene


def enhance(image):
    return skimage.exposure.adjust_sigmoid(image, cutoff=0.14, gain=12)

@click.command()
@click.argument("geojson_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("scene_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("outdir", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--view", default='nadir', help="Only visualize in some view, e.g. nadir, forward...")
@click.option("--padding", type=int, default=None, help="Padding around cropped images to include context")
@click.option("--enhance", is_flag=True, default=False, help="Perform brightness enhancement")
def main(geojson_file, scene_file, outdir, view, padding, enhance):
    """ This script visualizes 3d polygon (only) shapes in source images, e.g. nadir. This
    helps visualizing heights of objects, especially when those objects are not present in
    point clouds."""
    print "Reading scene file %s ..." % scene_file
    scene = Scene.read_yaml(scene_file)
    print "Reading data file %s ..." % geojson_file
    data = gps.read_file(geojson_file)

    # generate a mapping from images to all contained object TIDs

    print "Generating point index..."
    image_object_map = {}
    for _, item in data.iterrows():
        geometry = item.geometry
        for img_id in scene.find_overlapping_images(sg.box(*geometry.bounds), view):
            if img_id not in image_object_map:
                image_object_map[img_id] = []
            image_object_map[img_id].append(item.tid)
    print "Done."
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for img_id, tids in image_object_map.iteritems():
        cam = scene.get_camera(img_id)
        img = scene.read_image(img_id)
        img = np.rollaxis(img, 0, 3)

        for tid in tids:
            for idx, item in data[data.tid == tid].iterrows():
                if len(data[data.tid == tid]) > 1:
                    print "More than one item for TID %d" % tid

                if padding is not None:
                    vis_img = img.copy()
                else:
                    vis_img = img
                geometry = item.geometry
                category = item.category

                if isinstance(geometry, sg.Polygon):
                    coords = [geometry.exterior.coords]
                elif isinstance(geometry, sg.MultiPolygon):
                    coords = [g.exterior.coords for g in geometry.geoms]
                else:
                    continue

                print item.tid, img_id
                for coord in coords:
                    img_points = np.array([cam.world2img(point) for point in coord])
                    img_points = img_points.astype(np.int32)
                    rr, cc = polygon_perimeter(img_points[:, 1],
                                               img_points[:, 0],
                                               shape=cam.sensor_size[::-1])
                    vis_img[rr, cc] = 255
                    vis_img[rr, cc, 2] = 0
                if padding is not None:
                    patch = vis_img[img_points[:, 1].min() - padding : img_points[:, 1].max() + padding,
                                    img_points[:, 0].min() - padding : img_points[:, 0].max() + padding]
                    if enhance:
                        patch = enhance(patch)
                    if patch.size > 0:
                        outfile = os.path.join(outdir, "%s_%d_%d_%s.jpg" % (category, item.tid, idx, img_id))
                        sio.imsave(outfile, patch)
        if padding is None:
            if enhance:
                vis_img = enhance(vis_img)
            outfile = os.path.join(outdir, "%s_%s.jpg" % (category, img_id))
            sio.imsave(outfile, vis_img)

if __name__ == '__main__':
    main()

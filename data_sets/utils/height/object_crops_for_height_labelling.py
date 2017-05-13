import os.path

import click
import geopandas as gps
import numpy as np
import skimage.exposure as e
import skimage.io as sio

from objectdetection.oblique.scene import Scene


def object_hull(camera, polygon, min_height, max_height):
    p_lower = camera.w_shape2img(polygon, height=min_height)
    p_upper = camera.w_shape2img(polygon, height=max_height)
    polygon = p_lower.union(p_upper).convex_hull
    return polygon


def object_hull(camera, polygon, min_height, max_height):
    p_lower = camera.w_shape2img(polygon, height=min_height)
    p_upper = camera.w_shape2img(polygon, height=max_height)
    polygon = p_lower.union(p_upper).convex_hull
    return polygon


def crop_polygon(img, polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    min_x = int(round(min_x))
    min_y = int(round(min_y))
    max_x = int(round(max_x))
    max_y = int(round(max_y))
    return img[:, min_y:max_y, min_x:max_x], ((min_x, max_x), (min_y, max_y))


@click.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("scene_file", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path(exists=True, file_okay=False))
@click.argument("direction", type=str)
@click.argument("category_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--buffer-size", type=float, default=0.0)
@click.option("--object-height", type=float, default=4.0)
@click.option("--sigmoid", type=(float, float), default=(None, None))
def main(annotation_file, scene_file, outdir, direction, category_file, buffer_size,
         object_height, sigmoid):
    print "Reading annotations..."
    annotations = gps.read_file(annotation_file)
    with open(category_file) as fp:
        categories = fp.readlines()
    categories = [cat.strip() for cat in categories]
    annotations = annotations[annotations.category.isin(categories)]
    print "Found %d annotations (after filtering for categories)" % len(annotations)
    print "Loading scene..."
    scene = Scene.read_yaml(scene_file)
    view = scene.views[direction]

    for idx, t_sign in annotations.iterrows():
        tid = t_sign.tid
        category = t_sign.category
        print "Object TID %d (%s)" % (tid, category)
        t_sign = t_sign.geometry
        t_sign = t_sign.buffer(buffer_size)

        heights = scene.dsm.read_masked(t_sign)
        height = heights.min()

        matching_images = scene.find_overlapping_images(t_sign, view=direction)
        print "Found %d matching images" % len(matching_images)
        if len(matching_images) == 0:
            click.echo(click.style("Warning:" , fg='red') + " No matching image")
            continue

        # Find best image with largest angle.
        angles = []
        for i in matching_images:
            cam = view.get_camera(i)
          # calculating lower and upper polygon to approximate height extents.
            polygon = object_hull(cam, t_sign, height, height + object_height)
            angle = cam.angle_at(polygon.centroid)

            angles.append({"img_id": i, "angle" : angle})

        angles.sort(key=lambda x: x['angle'], reverse=True)
        img_id = angles[0]['img_id']


        cam = view.get_camera(img_id)
        polygon = object_hull(cam, t_sign, height, height + object_height)
        img = scene.read_image(img_id)
        cropped_object, window = crop_polygon(img, polygon)

        outimg = os.path.join(outdir, "%s_%s.jpg" % (tid, category))
        cropped_object = np.rollaxis(cropped_object, 0, 3)
        cutoff, gain = sigmoid
        if cutoff is not None:
            print "performing exposure correction..."
            cropped_object = e.adjust_sigmoid(cropped_object,
                                              cutoff=cutoff,
                                              gain=gain)
        if cropped_object.size > 0:
            sio.imsave(outimg, cropped_object)
        else:
            click.echo(click.style("Warning:", fg='red') + " Empty image " + img_id)


if __name__ == '__main__':
    main()

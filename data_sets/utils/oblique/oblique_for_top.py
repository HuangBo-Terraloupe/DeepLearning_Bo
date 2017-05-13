import os.path
from glob import glob

import click
import geopandas as gps
import rasterio as rio
import shapely.geometry as sg
import skimage.exposure as e
import skimage.io as sio
from skimage.transform import rescale

from objectdetection.oblique.camera import Camera
from objectdetection.utils.dsm import DSM


def glob_filetypes(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, "*.%s" % ext)))
    return files


def get_all_images(directory):
    filetypes = ['jpg', 'png', 'tif']
    return glob_filetypes(directory, filetypes)


@click.command()
@click.argument("top_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("oblique_footprints", type=click.Path())
@click.argument("dsm_file", type=click.Path())
@click.argument("ori_dir", type=click.Path())
@click.argument("oblique_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("top_footprint", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
@click.option("--dsm-heights", type=(float, float))
@click.option("--sigmoid", type=(float, float))
@click.option("--scale", type=float)
def main(top_dir, oblique_footprints, dsm_file, ori_dir, oblique_dir, top_footprint,
         outdir,
         dsm_heights, sigmoid, scale):
    """
    This tool takes a folder of (true-)ortho images and checks which of the given oblique images
    shows the same area. It was created to generate auxiliary oblique images
    """

    click.echo("True ortho images:\t%s" % top_dir)

    print "Reading %s for oblique footprints..." % oblique_footprints
    footprints = gps.read_file(oblique_footprints)
    top_footprint = gps.read_file(top_footprint).iloc[0].geometry

    dsm = DSM(dsm_file, min_height=dsm_heights[0], max_height=dsm_heights[1])

    for im_file in get_all_images(top_dir):
        filename = os.path.split(im_file)[-1]

        with rio.open(im_file) as ds:
            b = sg.box(*ds.bounds)
            b = b.intersection(top_footprint)

            oblique = footprints

            matches = oblique[oblique.contains(b)]
            if matches.shape[0] == 0:
                #                 print "Can only find intersecting oblique image..."
                intersection = oblique.intersection(b).area
                #                 print intersection.max()
                try:
                    best = oblique.loc[intersection[intersection > 0.0].argmax()]
                except:
                    print "No intersection found!"
                    continue
            else:
                best = matches.loc[matches.exterior.distance(b).argmax()]
                # best = matches.loc[matches.centroid.distance(b.centroid).argmin()]
            print "best forward oblique image is", best.image_id

            # FIXME: Make it image type agnostic
            oblique_file = os.path.join(oblique_dir, "%s.tif" % best.image_id)
            oblique_img = sio.imread(oblique_file)
            ori_file = os.path.join(ori_dir, '%s.ori' % best.image_id)
            cam = Camera.read_ori(ori_file)

            print "Area: %0.1f" % b.area
            points = list(b.buffer(-0.1).exterior.coords)[:-1]
            print "# points while buffering: %d --> %d" % (len(b.exterior.coords), len(points))

            # assert len(points) == len(b.exterior.coords) - 1

            img_points = []
            for point in points:
                height = dsm.height_at([point])[0]
                wp = point + (height,)
                img_p = cam.world2img(wp)
                img_points.append(img_p)
            img_poly = sg.Polygon(img_points)
            img_poly = img_poly.buffer(1.5)

            min_x, min_y, max_x, max_y = img_poly.bounds

            min_x = max(0, int(min_x))
            max_x = min(oblique_img.shape[1], int(max_x))

            min_y = max(0, int(min_y))
            max_y = min(oblique_img.shape[0], int(max_y))

            oblique_img = oblique_img[min_y:max_y, min_x:max_x]
            if oblique_img.size > 0:
                if scale is not None:
                    oblique_img = rescale(oblique_img, scale)
                if sigmoid is not None:
                    cutoff, gain = sigmoid
                    oblique_img = e.adjust_sigmoid(oblique_img,
                                                   cutoff=cutoff,
                                                   gain=gain)
                outfile = os.path.join(outdir, filename)
                print "%s --> %s" % (im_file, outfile)
                sio.imsave(outfile, oblique_img)



if __name__ == '__main__':
    main()
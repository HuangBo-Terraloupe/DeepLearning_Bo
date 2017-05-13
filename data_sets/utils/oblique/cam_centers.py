import os.path
from glob import glob

import click
import geopandas as gps

from objectdetection.oblique.camera import Camera


@click.command()
@click.argument("ori_dir")
@click.argument("outfile")
@click.argument("height", type=int)
@click.option("--filter", type=str)
def main(ori_dir, outfile, height, filter):
    """ This script calculates the center of each image in world coordinates
    and writes them to an GeoJSON file.
    """
    center_points = []
    if filter is not None:
        fname = "*%s*.ori" % filter
    else:
        fname = "*.ori"
    for ori_file in glob(os.path.join(ori_dir, fname)):
        cam = Camera.read_ori(ori_file)
        footprint = cam.footprint(height)
        img_id = os.path.split(ori_file)[-1].split('.')[-2]
        center_points.append({"image_id": img_id, "geometry": footprint.centroid})

    center_points = gps.GeoDataFrame(center_points)
    with open(outfile, 'w') as fp:
        fp.write(center_points.to_json())


if __name__ == '__main__':
    main()
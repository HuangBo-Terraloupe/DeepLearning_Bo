import click
import geopandas as gps
from glob import glob
import skimage.io as sio
from skimage.draw import polygon_perimeter, polygon
import os.path
import os


@click.command()
@click.argument("annotation-file", type=click.Path(dir_okay=False, exists=True))
@click.argument("images", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument("outdir", type=click.Path(file_okay=False))
def main(annotation_file, images, outdir):
    """ Rasterize annotations in images.
    """
    print "Opening %s ..." % annotation_file

    annotations = gps.read_file(annotation_file)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for im_file in glob(os.path.join(images, "*.jpg")):
        img_id = os.path.splitext(os.path.split(im_file)[-1])[0]
        im_annotations = annotations[annotations.image_id == img_id]

        img = sio.imread(im_file)
        print img.shape, img.min(), img.max(), img.mean()
        for idx, item in im_annotations.iterrows():
            category = item.category
            x_coords, y_coords = item.geometry.exterior.coords.xy
            cr, cc = polygon_perimeter(y_coords, x_coords, shape=img.shape[:2])

            if category == 'solar panel':
                r, g, b = 255, 255, 0
            elif category == 'glass':
                r, g, b = 255, 128, 0
            img[cr, cc, 0] = r
            img[cr, cc, 1] = g
            img[cr, cc, 2] = b

            cr, cc = polygon(y_coords, x_coords, shape=img.shape[:2])
            #
            # img[cr, cc, 0] = (img[cr, cc, 0] + r) / 2
            # img[cr, cc, 1] = (img[cr, cc, 1] + g) / 2
            img[cr, cc, 2] /= 4
        sio.imsave(os.path.join(outdir, "%s.jpg" % img_id), img)


if __name__ == '__main__':
    main()

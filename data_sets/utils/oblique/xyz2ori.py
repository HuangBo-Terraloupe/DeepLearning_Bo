import os
import os.path

import click

import objectdetection.oblique.ori_writer as ori_writer
from objectdetection import oblique as xyz


@click.command()
@click.argument("xyzfile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False))
@click.argument("focal_length", type=float)
@click.argument("pixel_size", type=float)
@click.option("--image-shape", type=(int, int))
@click.option("--principal-point", type=(float, float))
@click.option("--strip-prefix",
              type=int,
              help="Strip first N characters from image ID.",
              default=0)
def main(xyzfile, outdir, focal_length, pixel_size, image_shape, principal_point, strip_prefix):
    data = xyz.parse(xyzfile)
    image_ids = data.image_id.unique()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    sensor_size = (image_shape[1], image_shape[0])
    if principal_point is None:
        principal_point = (image_shape[1] / 2.0, image_shape[0] / 2.0)

    for image_id in image_ids:
        K, R, C = xyz.get_camera_parameters(data, image_id, focal_length,
                                            pixel_size, principal_point)
        image_id = image_id[strip_prefix:]
        print image_id

        ori_string = ori_writer.serialize(image_id, focal_length, pixel_size,
                                          sensor_size, principal_point, K, R,
                                          C)
        outfile = os.path.join(outdir, image_id + ".ori")
        with open(outfile, 'w') as fp:
            fp.write(ori_string)


if __name__ == '__main__':
    main()

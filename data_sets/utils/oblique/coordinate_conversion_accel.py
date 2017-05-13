import click
import numpy as np
import rasterio
from datetime import datetime
from numba import jit
from objectdetection.oblique.coordinate_conversion import read_dsm_within_bounds

from objectdetection.oblique.ori_parser import parse_ori

""" This module contains DSM ray tracing and needed  accelerated functions from
`coordinate_conversion.py `.
"""

now = datetime.now


@jit(nopython=True)
def __sample_raster__(position, raster, raster_transform, dsm_factor):
    """ Samples a given `raster` at the given positions after applying the
    `raster_transform`. Returns a numpy array containing the values at the
    given points.

    Args:
        position:         an array of geographical positions for which to
                          sample the given raster
        raster:           The raster to be sampled.
        raster_transform: a 6-tuple of the affine transform parameters.
        dsm_factor:       DSM value factor

    Returns:
        a numpy array with the sampled raster values.
    """
    sa, sb, sc, sd, se, sf = raster_transform
    vx = position[0]
    vy = position[1]

    cols = vx * sa + vy * sb + sc
    rows = vx * sd + vy * se + sf

    #instead of rounding, add 0.5 and clip decimals
    rows = (rows + 0.5).astype(np.int32)
    cols = (cols + 0.5).astype(np.int32)

    vals = np.zeros(cols.shape[0], dtype=np.float32)
    for i in range(rows.shape[0]):
        row = rows[i]
        col = cols[i]
        vals[i] = raster[row, col] * dsm_factor
    return vals


@jit(nopython=True)
def __project_dsm__(X,
                    C,
                    dsm,
                    dsm_transform,
                    max_height,
                    min_height,
                    height_threshold=0.0,
                    interval=1.0,
                    dsm_factor=1.0):
    ''' Projects a point on the image plane (given in world coordinates) to the
    DSM intersection point.'''
    direction = C - X
    x_start = (max_height - C[2]) / (C[2] - X[2])
    x_end = (min_height - C[2]) / (C[2] - X[2])

    x = np.arange(x_start, x_end + interval, interval)
    current_point = C.reshape((3, 1)) + x * direction.reshape((3, 1))
    # print "Shape of list of sample locations", current_point.shape
    dsm_heights = __sample_raster__(current_point[:2], dsm, dsm_transform, dsm_factor)

    above_ground = current_point[2] - dsm_heights
    below_ground = above_ground < height_threshold
    if below_ground.any():
        hit_loc = np.where(below_ground)[0][0]
        return current_point[:, hit_loc], dsm_heights, current_point[2]
    else:
        print("Warning! Did not find DSM intersection point!")
        return current_point[:, -1], dsm_heights, current_point[2]



@jit(nopython=True)
def __img2world__(X_img, K_inv, R_t, t):
    ''' Converts 2d image coordinates to world coordinates, given camera
    intrinsic and extrinsic parameters.

    Params:
    - X_img: input 2D image coordinates, e.g. [x, y] = [1300, 800]. Attention!
                 Order different from standard numpy ordering
    - K_inv: Inverted intrinsic camera matrix
    - R_t:   Transposed rotation matrix
    - t:     t = - R @ C

    Retuns:
    3D world coordinates of the image point. Unprojection to e.g. ground plane
    is not performed!
    '''

    x_world = np.dot(R_t, np.dot(K_inv, X_img) - t)
    return x_world
#
# height_image = ray_trace_dsm(wtrans, K_inv, R_t, C, t, dsm_part,
#                              max_height, min_height)


@jit(nopython=True)
def __ray_trace_dsm__(img_shape, K, R, C, dsm, dsm_transform, height_range,
                      height_threshold, interval, dsm_factor):
    '''
    '''

    K_inv = np.linalg.inv(K)
    t = -np.dot(R, C)
    R_t = R.T

    min_height = height_range[0]
    max_height = height_range[1]
    num_rows = img_shape[0]
    num_cols = img_shape[1]

    result = np.zeros((num_rows, num_cols), dtype=np.float64)

    point = np.ones((3, ), dtype=np.float64)
    for r in range(0, num_rows):
        for c in range(0, num_cols):
            point[0] = c
            point[1] = r
            wp = __img2world__(point, K_inv, R_t, t)

            ray_point, h1, h2 = __project_dsm__(wp, C, dsm, dsm_transform,
                                                max_height, min_height,
                                                height_threshold, interval,
                                                dsm_factor)
            result[r, c] = ray_point[2]
    return result


def ray_trace_dsm(img_shape, K, R, C, dsm, dsm_transform, height_range,
                  height_threshold, interval, dsm_factor):
    ''' Performs ray tracing of a given DSM to an oblique view. It generates an
    image of size `image_shape` of the values of the `dsm`. `K`, `R`, `C` are
    the extrinsic and intrinsic camera parameters.

    Args:
        img_shape (tuple(int, int)):  Shape of the generated image
        K (numpy.ndarray):            Intrinsic camera parameters
        R (numpy.ndarray):            Extrinsic camera parameters (rotation)
        C (numpy.ndarray):            Extrinsic camera parameter (translation)
        dsm (numpy.ndarray):          The DSM raster to be ray traced.
        dsm_transform (affine.Affine):The transform converting
        dsm_factor (float):           The factor by which DSM values are multiplied
    Returns:
        (numpy.ndarray):        A rendered DSM image from the perspective of the
                                given camera.
    '''

    return __ray_trace_dsm__(img_shape, K, R, C, dsm, dsm_transform[:6],
                             height_range, height_threshold, interval, dsm_factor)


@click.command()
@click.argument("oblique_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("ori_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("dsm_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("outfile", type=click.Path())
@click.option("--min-height", type=float, default=0.0)
@click.option("--max-height", type=float, default=1000.0)
@click.option(
    "--height_threshold",
    type=float,
    default=0.0,
    help=
    "Threshold at which height difference a DSM intersection is considered")
@click.option("--interval",
              type=float,
              default=0.4,
              help="The interval by which points are calculated")
@click.option("--dsm-factor", type=float, default=1.0, help='Factor for mapping DSM values to x,y,z units. E.g. 0.01 for Aschheim (int values).')
def main(oblique_file, ori_file, dsm_file, outfile, min_height, max_height,
         height_threshold, interval, dsm_factor):

    start = now()
    K, R, C, img_shape = parse_ori(ori_file)
    with rasterio.drivers():
        with rasterio.open(oblique_file) as ds:
            img_shape = [ds.profile['height'], ds.profile['width']]
            profile = ds.profile
        h_range = (min_height, max_height)

        dsm_part, dsm_transform = read_dsm_within_bounds(img_shape, dsm_file,
                                                         K, R, C, h_range)
        max_height = dsm_part.max() * dsm_factor
        h_range = (min_height, max_height)
        rendered_img = ray_trace_dsm(img_shape, K, R, C, dsm_part,
                                     dsm_transform, h_range, height_threshold,
                                     interval, dsm_factor)

        new_profile = {
            'count': 1,
            'driver': 'GTiff',
            'compress': 'deflate',
            'dtype': 'float32',
            'width': profile['width'],
            'height': profile['height']
        }

        with rasterio.open(outfile, 'w', **new_profile) as ds:
            ds.write(rendered_img.astype(np.float32), 1)

    print 'Processing took', now() - start


if __name__ == '__main__':
    main()

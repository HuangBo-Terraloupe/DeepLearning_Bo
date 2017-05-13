import numpy as np
import rasterio as rio
import rasterio.features as rf
from affine import Affine


class DSM(object):

    def __init__(self, dsm_file, ratio=1.0, min_height=0.0, max_height=1000.0):

        self.dsm_file = dsm_file
        self.ratio = ratio
        self.min_height = min_height
        self.max_height = max_height

        self.ds = rio.open(dsm_file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ds.close()

    def height_at(self, world_points):
        return list(self.ds.sample(world_points))

    def read_masked(self, polygon):
        """
        Reads all DSM values inside the given polygon.
        Args:
            polygon: A shapely.geometry.Polygon instance

        Returns:
            A `numpy.ma.maskedarray` of all valid values within the polygon.

        """
        geometry = polygon
        ul = self.ds.index(*geometry.bounds[0:2])
        lr = self.ds.index(*geometry.bounds[2:4])

        # read the subset of the data into a numpy array
        window = ((lr[0], ul[0] + 1), (ul[1], lr[1] + 1))
        data = self.ds.read(1, window=window)

        # create an affine transform for the subset data
        # FIXME check, if raster.window_transform can be used
        t = self.ds.affine
        shifted_affine = Affine(t.a, t.b, t.c + ul[1] * t.a, t.d, t.e, t.f + lr[0] * t.e)

        # rasterize the geometry
        mask = rf.rasterize(
            [(geometry, 0)],
            out_shape=data.shape,
            transform=shifted_affine,
            fill=1,
            all_touched=True,
            dtype=np.uint8)

        # create a masked numpy array
        masked_data = np.ma.array(data=data, mask=mask.astype(bool))
        # Filter NaN values
        masked_data = np.ma.masked_invalid(masked_data)
        return masked_data
import abc
import getpass

import datetime
import geopandas as gps
import numpy as np
import rasterio
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from scipy.optimize import leastsq
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon
from shapely.geos import TopologicalError
from shapely.ops import linemerge
from objectdetection.utils import logger
from objectdetection.utils.geometry.line_utils import LineConnector

from objectdetection.utils.io import gps_io_utils


class HeightNormalizer:
    '''
    Normalize the heights of objects of category along the width of the object.
    '''
    def __init__(self, data_category, input_file, out_filename, dsm_filename):
        self.META_DATA = {'creator' : getpass.getuser(), 'algorithm' : self.__class__.__name__,
                      'derived' : 'true', 'created_at' : str(datetime.datetime.now())}
        self.input_data = gps.read_file(input_file)
        self.category_data = self.input_data[self.input_data.category == data_category]
        self.category_data = self.category_data[self.category_data.geometry.type == 'Polygon']
        self.out_filename = out_filename
        self.features = [feature for _, feature in self.category_data.iterrows()]
        with rasterio.drivers():
            self.dsm = rasterio.open(dsm_filename)
        self.estimator = ModelBasedHeightEstimator1(self.dsm)
    
    def get_centerline(self, polygon):
        '''
        Returns the centerline for along the length of the polygon. It select the largest continous line 
        out of mulitple center lines.
        '''
        centerline = linemerge(Centerline(polygon, 0.08).createCenterline())
        line_connector = LineConnector()
        if centerline.type != 'LineString':
            centerline = line_connector.get_longest_continuous_line(centerline)
        return centerline
    
    def get_line_slices(self, feature, min_distance, buffer_distance):
        '''
        Returns the line slices perpendicular to the center line of the polygon. The number of returned
        line slices depend on the min_distance.
        @param feature: A polygon feature.
        @param min_distance: The distance between line slices.
        @param buffer_distance: The length of line slice which should be outside the polygon.
        @return: The list of line slices. Each line slice is a Line object with line end points. 
        '''
        polygon = feature.geometry
        centerline = self.get_centerline(polygon)
        ortho_line_slices = []
        total_points = int(centerline.length / min_distance) + 1
        last_point = centerline.coords[0]
        for i in xrange(1, total_points):
            slice_center = centerline.interpolate(min_distance * i)
            line = LineString([last_point, slice_center])
            line_slice = self.get_line_slice(polygon, line, slice_center, buffer_distance)
            if line_slice:
                ortho_line_slices.append(line_slice) 
            else:
                continue
            ortho_line_slices.append(line_slice)
            last_point = slice_center
        logger.info('Total line slices:{}, feature:{}'.format(len(ortho_line_slices), feature.id))
        return ortho_line_slices
    
    
    def get_line_slice(self, polygon, line, center, buffer_distance):
        '''
        Returns a line orthogonal to line which passes through center and extends by buffer_distance from polygon.
        '''
        linestring = self.get_ortho_line(line, center, 100)
        intersecting_line = polygon.intersection(linestring)
        if intersecting_line.type == 'MultiLineString':
            intersecting_line = max(intersecting_line, key=lambda x : x.length)
        elif intersecting_line.type == 'GeometryCollection':
            return None
        return self.extend_line(intersecting_line, buffer_distance)
    
    def get_ortho_line(self, line, center, length):
        center = np.mat(center.coords[0]).T
        p0, p1 = np.array(line.coords[0]), np.array(line.coords[-1])
        rot = np.mat([[0, -1], [1, 0]])
        u = rot * np.mat(p1 - p0).T / np.linalg.norm(p1 - p0)
        start_new, end_new = center - u * length, center + u * length
        return LineString([start_new.T.tolist()[0], end_new.T.tolist()[0]])
    
    def extend_line(self, line, buffer_distance):
        p0, p1 = np.array(line.coords[0]), np.array(line.coords[-1])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        start_new, end_new = p0 - u * buffer_distance, p1 + u * buffer_distance
        return LineString([start_new.T.tolist(), end_new.T.tolist()])

    def merge_polygons(self, new_polygon, old_polygon):
        old_polygon = old_polygon.buffer(0)
        new_polygon = new_polygon.buffer(0)
        merged_polygon = new_polygon.union(old_polygon)
        return merged_polygon

    def normalize_height(self):
        '''
        Normalizes and adds height data to features. It densifies the polygon by adding new points.
        '''
        normalized_features_map = {}
        failed_features = []
        for feature in self.features:
            logger.info('Normalizing heights for feature:{}'.format(feature.id))
            try:
                line_slices = self.get_line_slices(feature, 0.1, 0)
                new_points = []
                for line_slice in line_slices:
                    new_point = self.estimator.estimate_height(feature, line_slice)
                    new_points.append(new_point)
                new_poly_coords1 = [new_point[0] for new_point in new_points]
                new_poly_coords2 = [new_point[1] for new_point in new_points]
                new_poly_coords2.reverse()
                new_poly_coords = new_poly_coords1 + new_poly_coords2
                new_polygon = Polygon(new_poly_coords)
                old_polygon = feature.geometry
                merged_polygon = self.merge_polygons(new_polygon, old_polygon)
                normalized_features_map[feature.id] = merged_polygon
            except TopologicalError:
                logger.warn('Error while finding intersection for feature:{}'.format(feature.id))
                failed_features.append(feature.id)
                continue
        
        logger.info('Processed features:{}, Failed features:{}'.format(len(normalized_features_map), len(failed_features)))
        normalized_features_gs = []
        for feature_id, polygon in normalized_features_map.items():
            data = {'geometry' : polygon, 'source_features' : ','.join([feature_id])}
            data.update(self.META_DATA)
            normalized_features_gs.append(GeoSeries(data))
        normalized_gdf = GeoDataFrame(normalized_features_gs)
        gps_io_utils.write_gdf_json(normalized_gdf, self.out_filename)
        
class AbstractHeightEstimator:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, dsm):
        self.dsm = dsm
    
    @abc.abstractmethod
    def estimate_height(self, feature, line,):
        '''
        Should return two points on the polygon. Each point should have a height components as well.  
        '''
        raise NotImplementedError()
    
    def sampling(self, line, sample_per_meter, scale_factor):
        length = 0
        length_coord = []
        xy_on_line = []
        coords = []
        for pos in np.arange(0, line.length, line.length / sample_per_meter):
            length += line.length / sample_per_meter
            length_coord.append(length)
            p = line.interpolate(pos)
            coords.append(p.coords.xy)
            xy_on_line.append(np.array(p.coords.xy).reshape(2))
        height_values = self.get_height(self.dsm, xy_on_line, scale_factor)
        return length_coord, height_values, coords
    
    def get_height(self, ds, points_in_line, scale_factor):
        # Returns the height coordinate from the vrt file (which is opend as dsm) for every given coordinate
        with rasterio.drivers():
            heights = list(ds.sample(points_in_line))
        return np.array(heights, dtype=np.float64)[:, 0] / np.float(scale_factor)

class MeanBasedHeightEstimator(AbstractHeightEstimator):
    '''
    Removes outliers from the height values and return the average of remaining height values. 
    Any height which lies outside 1* standard deviation is considered outlier.
    '''    
    def __init__(self, dsm):
        super(MeanBasedHeightEstimator, self).__init__(dsm)
        
    def estimate_height(self, feature, line):
        _, height_values, __ = self.sampling(line, 33, 1)
        avg_height = np.average(height_values[abs(height_values - np.mean(height_values)) < 1 * np.std(height_values)])
        avg_height = (avg_height,)
        return [line.coords[0] + avg_height, line.coords[1] + avg_height]
    
    
class ModelBasedHeightEstimator(AbstractHeightEstimator):
    '''
    Model based height estimator which tries to fit the points along a line by fitting a box.
    '''
    
    def __init__(self, dsm):
        super(ModelBasedHeightEstimator, self).__init__(dsm)
    
    def estimate_height(self, feature, line):
        AbstractHeightEstimator.estimate_height(self, feature, line)
    
    def rectangular_function2(self, X, *p0):
        (start, height, stop, base) = p0
        temp = np.zeros_like(X)
        for i, x in enumerate(X):
            if (x < start): temp[i] = base
            if (start <= x) and (x <= stop): temp[i] = height
            if (stop < x): temp[i] = base
        return temp

    def rectangular_function(self, X, *p0):
        (start, stop, height, m1, t1, m2, t2) = p0
        temp = np.zeros_like(X)
        for i, x in enumerate(X):
            if (x < start):
                temp[i] = m1 * x + t1
            elif (start <= x) and (x <= stop):
                temp[i] = height
            elif (x > stop):
                temp[i] = m2 * x + t2
            else:
                raise Exception('Error')
        return temp

    def fit(self, funk, x, y, coo, p0):
        if p0 == [0, 0, 0, 0]:
            p0 = [0.2, 0.9, np.max(y), y[0]]
        errfunc = lambda p0, x, y: self.rectangular_function2(x, *p0) - y
        p1, success = leastsq(errfunc, p0, args=(x, y), epsfcn=1)
        error = np.array(x).shape[0] * np.sum(errfunc(p1, x, y) ** 2)
        start, stop, height = p1[:3]
        eval = funk(x, *p1)
        equal_indices = []
        eps = 0.1
        for i, val in enumerate(eval):
            if abs(val - height) < eps:
                equal_indices.append(i)
        temp = np.where(equal_indices)[0]
        logger.info('eval:{}, p1:{}, temp:{}'.format(funk(x, *p1), p1, temp))
        return coo[temp[0]], coo[temp[-1]], height, error, p1

    def fit_line(self, feature, line, p0):
        '''
        Returns two 2d and the corresponding height value
        '''
        X, heights, coords = self.sampling(line, 33, 100)
        point1, point2, h, error, p1 = self.fit(self.rectangular_function2, X, heights, coords, [0, 0, 0, 0])
        return point1, point2, h, error, p1
    
class ModelBasedHeightEstimator1(ModelBasedHeightEstimator):

    def __init__(self, dsm):
        super(ModelBasedHeightEstimator, self).__init__(dsm)
        
    def estimate_height(self, feature, line):
        x, y, coo = self.sampling(line, 33, 100)
        funk = self.rectangular_function2
        THL = 1
        p0 = [0.2, 0.9, np.max(y), y[0]]
        errfunc = lambda p0, x, y: self.rectangular_function2(x, *p0) - y
        p1, success = leastsq(errfunc, p0, args=(x, y), epsfcn=1)
        error = np.array(x).shape[0] * np.sum(errfunc(p1, x, y) ** 2)
        if 0 > error and error > THL:
            return self.delagator.estimate_height(feature, line)
        else:
            start, stop, height, _ = p1
            eval = funk(x, *p1)
            equal_indices = []
            eps = 0.1
            temp = np.where(eval == height)
        
            point1 = coo[temp[0]]
            point2 = coo[temp[-1]]
        
            point1_x_y_z = [point1[0], point1[1], height]
            point2_x_y_z = [point2[0], point2[1], height]
            return point1_x_y_z, point2_x_y_z

if __name__ == '__main__':
    
    height_normalizer = HeightNormalizer('guardrails', '/home/sanjeev/workspaces/shapes/aschheim/gt_labels.geojson',
                                         '../data/poly11_normalized.geojson', '/home/sanjeev/image_data/dsm.vrt')
    height_normalizer.normalize_height()

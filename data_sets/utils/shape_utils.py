import geopandas as gps
import numpy as np
from intervaltree.intervaltree import IntervalTree
from networkx.utils.union_find import UnionFind
from rtree import index
from shapely.geometry import box
from shapely.geos import TopologicalError
from skimage.transform import resize
from objectdetection.utils import logger

np.random.seed(42)


def get_dimensions(geodataframe):
    ''' Appends dimensions of bounds (width and height) to input dataframe'''
    bounds = geodataframe.bounds
    geodataframe['width'] = (bounds.maxx - bounds.minx).astype(np.int32)
    geodataframe['height'] = (bounds.maxy - bounds.miny).astype(np.int32)
    return geodataframe


def random_box(img_bounds, target_size):
    ''' Generates a random square bounding box within the given image bounds.
    '''
    x = np.random.randint(0, img_bounds.bounds[2] - target_size)
    y = np.random.randint(0, img_bounds.bounds[3] - target_size)
    new_sample = box(x, y, x + target_size, y + target_size)
    assert new_sample.within(img_bounds)
    return new_sample


def get_non_overlapping(negative_ratio, existing, img_bounds):
    n_negative_samples = negative_ratio * len(existing)
    negative_samples = []

    target_size = int(get_dimensions(existing.copy()).width.mean())

    for i in range(n_negative_samples):
        valid = False
        while not valid:
            new_sample = random_box(img_bounds, target_size)
            valid = not existing.overlaps(new_sample).any()
        negative_samples.append(new_sample)
    return negative_samples


def square_patches(geodataframe, img, size, pad, negative_ratio=3):
    ''' returns a numpy array of squares patches for the given GeoDataframe.'''
    df = get_dimensions(geodataframe)

    img_bounds = box(0, 0, img.shape[2], img.shape[1])

    dst_size = size + int(size * 2 * pad)
    dst_size = (img.shape[0], dst_size, dst_size)

    # Padding coordinates to make square boxes.
    bounds = df.bounds
    square_sizes = np.maximum(df.width, df.height)

    extra_pad = (pad * square_sizes).astype(np.int32)

    height_pad = square_sizes - df.height
    top_pad = (height_pad / 2).astype(np.int32) + extra_pad
    bottom_pad = top_pad + (height_pad % 2) + extra_pad

    width_pad = square_sizes - df.width
    left_pad = (width_pad / 2).astype(np.int32) + extra_pad
    right_pad = left_pad + (width_pad % 2) + extra_pad

    bounds.minx -= left_pad
    bounds.maxx += right_pad
    bounds.miny -= top_pad
    bounds.maxy += bottom_pad
    print "Image shape", img.shape
    bounds = bounds.astype(np.int32)

    img_bounds = box(0, 0, img.shape[2], img.shape[1])

    positive_samples = []
    for _, b in bounds.iterrows():
        positive_samples.append(box(b.minx, b.miny, b.maxx, b.maxy))
    positive_samples = gps.GeoSeries(positive_samples)

    print "Generating negative samples..."

    negative_samples = get_non_overlapping(negative_ratio, positive_samples,
                                           img_bounds)
    negative_samples = gps.GeoSeries(negative_samples)
    all_samples = positive_samples.append(negative_samples)

    labels = df.label.tolist() + [0, ] * len(negative_samples)
    print "Generating %d samples (positive + negative)" % len(labels)
    patches = []
    valid_labels = []

    print "Cropping and resizing patches..."
    for label, poly_box in zip(labels, all_samples):
        if poly_box.within(img_bounds):
            minx, miny, maxx, maxy = poly_box.bounds
            minx = int(minx)
            miny = int(miny)
            maxx = int(maxx)
            maxy = int(maxy)
            patch = img[:, miny:maxy, minx:maxx]
            patch = resize(patch, (dst_size))
            patches.append(patch)
            valid_labels.append(label)
        else:
            print "Box not completely within image bounds:", poly_box
    patches = np.array(patches)
    labels = np.array(valid_labels)
    return patches, labels

def get_neighbouring_shapes(shape_features, boundary=0):
    '''
    For each shape(line or polygon), returns the set of shapes which could possibly intersect/overlap with it. 
    The shapes in the vicinity of a shape are found by the following algorithm  
    1. Create a rectangle for each shape by its bounds. 
    2. Build the horizontal and vertical interval trees for (minx, maxx) and (miny, maxy) intervals for all rectangles.
    3. For each rectangle search the overlapping horizontal and vertical intervals.
    4. The rectangles which are both in horizontal and vertical overlapping intervals is overlapping with current rectangle
    and is a possible candidate for intersecting/overlapping shapes.
    5. Build a map of such neighbouring shapes for each shape.
    
    Complexity: O(nlogn + n*m) where 
     n = total number of shapes, m = average number of shapes in the vicinity
    '''
    class Rectangle:
        def __init__(self, uid, shape, boundary=0):
            self.uid = uid
            self.minx, self.miny, self.maxx, self.maxy = shape.bounds
            self.minx -= boundary
            self.maxx += boundary
            self.miny -= boundary
            self.maxy += boundary
        
        def __repr__(self):
            return "Rectangle({}): {},{},{},{}".format(self.uid, self.minx, self.miny, self.maxx, self.maxy)
            
    rectangles = [Rectangle(feature.id, feature.geometry, boundary) for feature in shape_features]
    horizontal_tree = IntervalTree()
    vertical_tree = IntervalTree()
    for rectangle in rectangles:
        horizontal_tree.addi(rectangle.minx, rectangle.maxx, rectangle)
        vertical_tree.addi(rectangle.miny, rectangle.maxy, rectangle)
    
    feature_vs_nearest_features = {}

    for rectangle in rectangles:
        overlapping_rectanglesx = horizontal_tree.search(rectangle.minx, rectangle.maxx)
        overlapping_rectanglesy = vertical_tree.search(rectangle.miny, rectangle.maxy)
        overlapping_setx = {interval.data.uid for interval in overlapping_rectanglesx}
        overlapping_sety = {interval.data.uid for interval in overlapping_rectanglesy}
        feature_vs_nearest_features[rectangle.uid] = overlapping_setx.intersection(overlapping_sety)
    return feature_vs_nearest_features


def get_overlapping_polygons(features, threshold, bbuffer=0):
    '''
    For each polygon feature returns the list of features which have common area greater than threshold.
    @param features: A list of features(GeoSeries) objects.
    @param threshold: Threshold for common area fraction.
    @param bbuffer: Buffer to be added around each shape while finding overlapping shapes so that 
    nearest shapes also be considered as overlapping candidates.
    @return: overlapping_polygon_map, feature id vs list of overlapping feature ids. 
    '''
    overlapping_set = UnionFind()
    intersecting_set = UnionFind()
    feature_map = {feature.id: feature for feature in features}
    feature_vs_nearest_features = get_neighbouring_shapes(features, boundary=bbuffer)
    failed_feature_pairs = set()
    for feature1 in features:
        polygon1 = feature1.geometry.buffer(bbuffer, join_style=2)
        for feature2_id in feature_vs_nearest_features[feature1.id]:
            if feature1.id != feature2_id:
                feature2 = feature_map[feature2_id]
                polygon2 = feature2.geometry.buffer(bbuffer, join_style=2)
                try:
                    union_area = polygon1.union(polygon2).area.real
                    intersect_area = polygon1.intersection(polygon2).area.real
                    intersect_area_fr = intersect_area / union_area
                    if intersect_area_fr > threshold:  # overlapping polygons ...
                        overlapping_set.union(overlapping_set[feature1.id], overlapping_set[feature2.id])
                    elif intersect_area_fr > 0:  # intersecting polygons...
                        intersecting_set.union(intersecting_set[feature1.id], intersecting_set[feature2.id])
                except TopologicalError:
                    failed_feature_pairs.add((feature1.id, feature2_id))
                    logger.error('Self intersecting polygons. Cannot find union for polygons:{} and {}'.
                                 format(feature1.id, feature2.id))
    overlapping_polygon_map = get_disjoin_sets(overlapping_set)
    intersecting_polygon_map = get_disjoin_sets(intersecting_set)
    logger.info('Total polygons before merging {}, after merging:{}, intersecting:{}, failed pairs:{}'
                .format(len(features), len(overlapping_polygon_map), len(intersecting_polygon_map), len(failed_feature_pairs)))
    return overlapping_polygon_map, intersecting_polygon_map, failed_feature_pairs

def get_disjoin_sets(union_find):
    '''
    Iterates over all sets in UnionFind and returns a dict of parent of each set versus the set.
    @param union_find: networkx.utilities.union_find.UnionFind object.
    '''
    disjoint_sets = {}
    for item in union_find:
        parent = union_find[item]
        if parent not in disjoint_sets:
            disjoint_sets[parent] = []
        disjoint_sets[parent].append(item)
        
    return disjoint_sets

def get_merged_featgeo(features, merged_feature_map, bbuffer=0):
    '''
    Merges(union) each polygon feature geometry of merged_feature_map with all corresponding features 
    found in merged_feature_map and creates a new feature collection.
    @param features: List of shape features.
    @param merged_feature_map: parent id vs list of feature ids to be merged.
    @param bbuffer: Buffer to by added around each shape feature while taking union so that 
    the merged features look smooth.    
    '''
    feature_map = {feature.id: feature for feature in features}
    new_feature_collection = {}
    failed_features = set()
    for parent_id, merged_feature_idxs in merged_feature_map.iteritems():
        gs_union = feature_map[merged_feature_idxs[0]].geometry.buffer(bbuffer, join_style=2)
        for i in xrange(1, len(merged_feature_idxs)):
            gs = feature_map[merged_feature_idxs[i]].geometry.buffer(bbuffer, join_style=2)
            gs_union = gs_union.union(gs)
        gs_union = gs_union.buffer(-bbuffer, join_style=2)
        # breaks in case union returns a collection of polygon and line...
        if gs_union.type == 'GeometryCollection':
            failed_features.update(merged_feature_idxs)
        else:
            new_feature_collection[parent_id] = gs_union
    return new_feature_collection, failed_features

def all_nearest_points_2d(source_points, target_points):
    '''
    Returns the a list which contains for each point from source_points the index of nearest point from 
    the target_points.
    @param source_points: List of (x,y) source points. 
    @param target_points: List of (x,y) points which define the search space.
    @return: A list of (source_point, target_point_indx). 
    '''
    def normalize_bound(point, buff=1e-09):
        minx, maxx = point[0] - buff, point[0] + buff
        miny, maxy = point[1] - buff, point[1] + buff
        return minx, miny, maxx , maxy
    
    # Build rtree for all target 2d-points...
    idx = index.Index()
    for i, point in enumerate(target_points):
        idx.add(i, normalize_bound(point), point)
    
    source_points_vs_nearest = []
    for point in source_points:
        nearest_points = list(idx.nearest(normalize_bound(point), objects=True))
        source_points_vs_nearest.append((point, nearest_points[0].id))
    
    return source_points_vs_nearest

import json
import geopandas as gpd

from shapely.geometry import polygon, Point
from geopandas import GeoDataFrame

def write_gdf_json(geo_df, out_filename, epsg_out=32632, should_format=False):
    """ Function to save (geo)data as *.geojson (needs to have .crs information!)
    Args:
        geo_df: GeoDataFrame in memory
        out_filename: full output path for *.geojson file
        epsg_out: If EPSG is not mentioned in the input GeoDataFrame, then this EPSG will be used.
        should_format: boolean that allows for readable indentation in the output json
    Returns:
        *.geojson file in output directory based on EPSG of vector data

    """

    # build crs header information to the header (easier I/O)
    if geo_df.crs is not None:
        epsg_out = int(geo_df.crs['init'].replace('epsg:', ''))

    header = {
        'type': 'name',
        'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % epsg_out}
    }

    # add header to dictionary
    result = json.loads(geo_df.to_json())
    result['crs'] = header

    # indent based on formatting decision
    if should_format:
        indent = 4
    else:
        indent = None

    # save as *.geojson
    with open(out_filename, 'w') as fp:
        json.dump(result, fp, indent=indent, sort_keys=True)

def find_entry_points_by_remove_the_diffRoads(parking_geojson, road_geosjon, output_file):
    '''find all the roads in which intersects with the parkings, remove the roads whos both sides intersects with
    parkings. the remain points will be entry and exist points.
    Args:
        parking_geojson:
        road_geosjon:
        output_file:

    Returns:

    '''
    # load road and parking geometries
    osm_roads = gpd.read_file(road_geosjon)
    parkings = gpd.read_file(parking_geojson)
    print('loading files are done !')

    # finding the entry and exit points
    save_geometry = {'geometry': []}
    spatial_index_roads = osm_roads.sindex
    print('creating spatial indexes are done !')

    for index, row in parkings.iterrows():
        parking_geo = row.geometry
        possible_matches_index = list(spatial_index_roads.intersection(parking_geo.bounds))
        possible_matches = osm_roads.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(parking_geo)]

        # loop through all the roads which intersetcs with the parking
        for i in range(len(precise_matches)):
            road_diff_ml = precise_matches.geometry.iloc[i].difference(polygon.LinearRing(list(parking_geo.exterior.coords)))

            # for each road which intersects with parking, the difference are multi-line-strings, loop through the ml
            # and drop the line-string which both sides interstcts with parking
            print(str(type(road_diff_ml)))
            try:
                for line in road_diff_ml:
                    intersection_info = [Point(line.coords[0]).intersects(parking_geo.buffer(0.000001)),
                                         Point(line.coords[-1]).intersects(parking_geo.buffer(0.00001))]
                    side_points = [Point(line.coords[0]), Point(line.coords[1])]
                    if intersection_info.count(True) == 1:
                        save_geometry['geometry'].append(side_points[intersection_info.index(True)])
            except:
                intersection_info = [Point(road_diff_ml.coords[0]).intersects(parking_geo.buffer(0.000001)),
                                     Point(road_diff_ml.coords[-1]).intersects(parking_geo.buffer(0.00001))]
                side_points = [Point(road_diff_ml.coords[0]), Point(road_diff_ml.coords[1])]
                if intersection_info.count(True) == 1:
                    save_geometry['geometry'].append(side_points[intersection_info.index(True)])


        if index % 1000 == 0:
            print(index)

    # save the exit points into a single geojson file
    df = GeoDataFrame(save_geometry)

    # remove duplications
    G = df["geometry"].apply(lambda geom: geom.wkb)
    df = df.loc[G.drop_duplicates().index]

    # set crs of the dataframe
    df.crs = {'init': 'epsg:32632', 'no_defs': True}

    write_gdf_json(df, out_filename=output_file,
                   epsg_out=32632, should_format=False)



def run_calculation_intersection(parking_geojson, road_geosjon, output_file):
    '''Finding the entry and exist points by finding the intersection-points of roads and polygons
    Args:
        parking_geojson:
        road_geosjon:
        output_file:

    Returns:

    '''

    # load road and parking geometries
    osm_roads = gpd.read_file(road_geosjon)
    parkings = gpd.read_file(parking_geojson)
    print('loading files are done !')

    # finding the entry and exit points
    id = 0
        
    save_geometry = {'geometry': []}
    spatial_index_roads = osm_roads.sindex
    print('creating spatial indexes are done !')

    for _, row in parkings.iterrows():

        possible_matches_index = list(spatial_index_roads.intersection(row.geometry.bounds))
        possible_matches = osm_roads.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(row.geometry)]
        if len(precise_matches) > 0:
            print('found parking intersects with roads:', len(precise_matches))
            parking_line_ring = polygon.LinearRing(list(row.geometry.exterior.coords))

            for i in range(len(precise_matches)):
                point = precise_matches.geometry.iloc[i].intersection(polygon.LineString(parking_line_ring))
                if str(point) != 'GEOMETRYCOLLECTION EMPTY':
                    save_geometry['geometry'].append(point)

        if id % 100 == 0:
            print(id)
        id = id + 1


    # save the exit points into a single geojson file
    df = GeoDataFrame(save_geometry)

    # remove duplications
    G = df["geometry"].apply(lambda geom: geom.wkb)
    df = df.loc[G.drop_duplicates().index]

    # set crs of the dataframe
    df.crs = {'init': 'epsg:32632', 'no_defs': True}


    write_gdf_json(df, out_filename=output_file,
                   epsg_out=32632, should_format=False)



if __name__ == '__main__':
    parking_geojson = '/home/terraloupe/Dataset/germany_parkings/Exit_Entry_Point/approach_v4/parkings.geojson'
    road_geosjon = '/home/terraloupe/Dataset/germany_parkings/Exit_Entry_Point/approach_v4/roads.geojson'
    output_file = '/home/terraloupe/Dataset/germany_parkings/Exit_Entry_Point/approach_v4/points.geojson'
    find_entry_points_by_remove_the_diffRoads(parking_geojson, road_geosjon, output_file)
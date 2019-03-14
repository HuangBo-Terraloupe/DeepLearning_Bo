import json
import geopandas as gpd

from geopandas import GeoDataFrame
from shapely.ops import cascaded_union

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


def run(input_geojson, output_file):

    # read input geojson
    buffer_points = gpd.read_file(input_geojson)
    spatial_index = buffer_points.sindex

    # init geo dataframe
    union_points = {'geometry': []}
    already_merge_geo = []

    for index, row in buffer_points.iterrows():

        if index % 1000 == 0:
            print(index)

        possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))
        possible_matches = buffer_points.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(row.geometry)]
        if len(precise_matches) > 1:
            if index in already_merge_geo:
                continue
            else:
                intersections = precise_matches.geometry.intersection(row.geometry)
                union = cascaded_union(intersections)
                #union = intersections.unary_union
                union_points['geometry'].append(union)

                index_list = list(intersections.index)
                #index_list.remove(index)
                already_merge_geo = already_merge_geo + index_list
        else:
            if index in already_merge_geo:
                continue
            else:
                union_points['geometry'].append(row.geometry)

    df = GeoDataFrame(union_points)

    # remove duplications
    G = df["geometry"].apply(lambda geom: geom.wkb)
    df = df.loc[G.drop_duplicates().index]

    # set crs of the dataframe
    df.crs = {'init': 'epsg:32632', 'no_defs': True}


    write_gdf_json(df, out_filename=output_file, epsg_out=32632, should_format=False)

if __name__ == '__main__':
    input_geojson = '/home/terraloupe/Dataset/germany_parkings/Exit_Entry_Point/approach_v2_buffer/buffered_parking.geojson'
    output_file = '/home/terraloupe/Dataset/germany_parkings/Exit_Entry_Point/approach_v2_buffer/buffered_parking_combined_v2.geojson'
    run(input_geojson, output_file)
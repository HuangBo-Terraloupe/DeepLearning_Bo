import geopandas as gpd

from geopandas import GeoDataFrame
from shapely.ops import cascaded_union
from utils.geometry import write_gdf_json

def run(input_geojson, output_file):

    # read input geojson
    df = gpd.read_file(input_geojson)
    spatial_index = df.sindex

    # init geo dataframe
    union_df = {'geometry': []}
    already_merge_geo = []

    for index, row in df.iterrows():

        if index % 1000 == 0:
            print(index)

        possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))
        possible_matches = df.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(row.geometry)]
        if len(precise_matches) > 1:
            if index in already_merge_geo:
                continue
            else:
                intersections = precise_matches.geometry.intersection(row.geometry)
                union = cascaded_union(intersections)
                union_points['geometry'].append(union)

                index_list = list(intersections.index)
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
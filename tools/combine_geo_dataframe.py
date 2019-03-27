import pandas as pd
import geopandas as gpd

from glob import glob
from utils.geometry import explode

def combine_geo_dataframe(input_folder, input_extension, output_file, crs=32632):
    '''
    Args:
        input_folder:
        input_extension:
        output_folder:
        crs:

    Returns:

    '''
    input_files = glob(input_folder + '/*.' + input_extension)
    print('the number of input files is:', len(input_files))
    df = []

    for id, input_file in enumerate(input_files):

        print(id)
        # read data frame
        df_temp = gpd.read_file(input_file)

        # split the geometry into multi-parts
        df_temp = explode(df_temp)

        df.append(df_temp)


    df = pd.concat(df)
    df = gpd.GeoDataFrame(df)

    # set dataframe crs
    #df.crs = {'init': 'epsg:' + str(crs), 'no_defs': True}
    df.crs = df_temp.crs
    print(df.crs)


    if output_file.endswith('.geojson'):
        df.to_file(output_file, driver='GeoJSON')
    elif output_file.endswith('.gpkg'):
        df.to_file(output_file, driver='GPKG')
    elif output_file.endswith('.shp'):
        df.to_file(output_file, driver='ESRI Shapefile')
    else:
        print('unknown dataformat, the GeoDataframe is not saved！')


if __name__ == '__main__':
    input_folder = '/home/terraloupe/Dataset/brandenburg_berlin_road_area/vectors'
    input_extension = 'shp'
    output_file = '/home/terraloupe/Dataset/brandenburg_berlin_road_area/combine_road.shp'
    combine_geo_dataframe(input_folder, input_extension, output_file, crs=32632)
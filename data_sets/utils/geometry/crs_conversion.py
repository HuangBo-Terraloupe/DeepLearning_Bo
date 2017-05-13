import json
import os

import click
import geopandas as gps
from fiona.crs import from_epsg


def crs_transform(df, src_crs, out_crs):
    """
    Converts GeoPandas GeoDataFrame between different coordinate reference systems (crs)
    :param df: GeoDataFrame in source crs
    :param src_crs: source crs
    :param out_crs: output crs
    :return: GeoDataFrame in output crs
    """
    df.crs = from_epsg(src_crs)

    if src_crs != out_crs:
        df_out = df.to_crs(from_epsg(out_crs))
    else:
        df_out = df

    return df_out


def crs_conversion(input_file, out_file, src_crs=4326, out_crs=32632, only_coords=False):
    """
    Loads either *.shp or *.geojson and converts crs and saves as *.geojson.
    Optionally allows to "slim" output by removing properties from each feature -> only saving coordinates.
    :param input_file: *.shp or *.geojson file
    :param out_file: desired output *.geojson file
    :param src_crs: source/input epsg code
    :param out_crs: output epsg code
    :param only_coords: flag to "slim" output with only geometry
    :return: *.geojson of input file in out_crs
    """
    # ensure file is either *.shp or *.geojson
    _, file_ext = os.path.splitext(input_file)
    valid_ext = ['.shp', '.geojson']
    assert any([file_ext == v for v in valid_ext])

    # read input file
    data = gps.read_file(input_file)

    # convert to desired projection (if needed)
    result = crs_transform(data, src_crs, out_crs)

    # allow "slim" output with only coordinates
    if only_coords:
        result = result['geometry']

    # build crs header information to the header (easier I/O)
    header = {
        'type': 'name',
        'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % out_crs}
    }

    # add header to dictionary
    result = json.loads(result.to_json())
    result['crs'] = header

    # save rest to *.geojson
    with open(out_file, 'w') as fp:
        json.dump(result, fp)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_file', type=click.Path(exists=True, dir_okay=False))
@click.option("--src_crs", type=int, default=4326)
@click.option("--out_crs", type=int, default=32632)
@click.option("--only_coords", is_flag=True, default=False)
def main(input_file, out_file, src_crs, out_crs, only_coords):
    crs_conversion(input_file, out_file, src_crs, out_crs, only_coords)


if __name__ == '__main__':
    main()
from fiona.crs import from_epsg

import geopandas as gps
import rasterio
from shapely import affinity
import click


@click.command()
@click.argument('infile', type=click.Path(exists=True, dir_okay=False))
@click.argument('image_dir',
                type=click.Path(exists=True,
                                dir_okay=True,
                                file_okay=False))
@click.argument('outfile', type=click.Path())
@click.option('--image-type', default='tif')
@click.option('--image-crs', type=int, default=31468)
def main(infile, image_dir, outfile, image_type, image_crs):
    ''' Transforms feature coordinates that are given in image coordinates to
    WGS-84 coordinates.
    '''

    print "Opening %s" % infile
    data = gps.read_file(infile)

    source_crs = from_epsg(image_crs)
    target_crs = from_epsg(4326)

    transform_parameters = {}
    with rasterio.drivers():
        for img_id in data.image_id.unique():
            print img_id
            img_file = "%s/%s.%s" % (image_dir, img_id, image_type)
            with rasterio.open(img_file) as ds:
                tf = ds.transform
                aff_transform = tf[1:3] + tf[4:] + [tf[0]] + [tf[3]]
                transform_parameters[img_id] = aff_transform

    transformed_shapes = []

    for _, item in data.iterrows():
        aff = transform_parameters[item.image_id]
        transformed = affinity.affine_transform(item.geometry, aff)
        transformed_shapes.append(transformed)

    data.geometry = gps.GeoSeries(transformed_shapes, crs=source_crs)
    data_wgs84 = data.to_crs(target_crs)

    with open(outfile, 'w') as fp:
        fp.write(data_wgs84.to_json(indent=0))


if __name__ == '__main__':
    main()

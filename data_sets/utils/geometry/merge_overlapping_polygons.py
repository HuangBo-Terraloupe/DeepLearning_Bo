import click
from shapely.geometry import MultiPolygon
import geopandas as gps
import pandas as ps
from fiona.crs import from_epsg


@click.command()
@click.argument('infile', type=click.Path(exists=True))
@click.argument('outfile', type=click.Path())
@click.option('--buffer-distance', type=float, default=0.5)
def main(infile, outfile, buffer_distance):
    source_crs = from_epsg(31468)
    target_crs = from_epsg(4326)
    print "Reading %s..." % infile
    data = gps.read_file(infile)
    # data.crs = target_crs
    # data = data.to_crs(source_crs)
    results = []
    print "%d objects found" % data.shape[0]

    for cat in data.category.unique():
        print "Merging category %s..." % cat
        cat_data = data[data.category == cat]

        polygons = MultiPolygon(list(cat_data.geometry))
        polygons = polygons.buffer(buffer_distance, join_style=2, cap_style=2)
        polygons = polygons.buffer(-buffer_distance, join_style=2, cap_style=2)
        polygons = gps.GeoSeries([p for p in polygons.geoms])

        result = gps.GeoDataFrame({'category': cat, 'geometry': polygons})
        results.append(result)

    results = ps.concat(results)
    results = gps.GeoDataFrame(results)
    print "%d objects generated" % results.shape[0]
    print "Writing %s." % outfile

    print results.groupby('category').count()

    with open(outfile, 'w') as fp:
        fp.write(results.to_json())

if __name__ == '__main__':
    main()

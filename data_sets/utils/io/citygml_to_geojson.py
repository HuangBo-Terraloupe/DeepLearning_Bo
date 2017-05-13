import click
import xml.etree.ElementTree as Tree
from shapely.geometry import Polygon
from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame
from glob import glob


def get_polygon_from_text(polygon_text):

    values = polygon_text.split(" ")
    n_points = len(values) / 3

    points = []
    for i in range(n_points):
        x = float(values[i*3 + 0])
        y = float(values[i*3 + 1])
        points.append([x, y])

    return Polygon(points)


def citygml_to_geojson(input_dir, outfile):

    files = glob( input_dir + '*.gml')
    gs_features = []
    for file_path in files:
        xml_data = Tree.parse(file_path)
        roof_surfaces = xml_data.getroot().findall(
            '{http://www.opengis.net/citygml/1.0}cityObjectMember/'
            '{http://www.opengis.net/citygml/building/1.0}Building/'
            '{http://www.opengis.net/citygml/building/1.0}boundedBy/'
            '{http://www.opengis.net/citygml/building/1.0}RoofSurface')

        for roofs in roof_surfaces:

            information = {}
            id = roofs.attrib['{http://www.opengis.net/gml}id']
            information['id'] = id
            information['type'] = 'RoofSurface'

            info = roofs.findall('{http://www.opengis.net/citygml/generics/1.0}stringAttribute')

            for child in info:
                name = child.attrib['name']
                value = child.find('{http://www.opengis.net/citygml/generics/1.0}value').text
                information[name] = value

            polygon_text_list = roofs.findall('{http://www.opengis.net/citygml/building/1.0}lod2MultiSurface/'
                                              '{http://www.opengis.net/gml}MultiSurface/'
                                              '{http://www.opengis.net/gml}surfaceMember/'
                                              '{http://www.opengis.net/gml}Polygon/'
                                              '{http://www.opengis.net/gml}exterior/'
                                              '{http://www.opengis.net/gml}LinearRing/'
                                              '{http://www.opengis.net/gml}posList')

            for idx, polygon_text in enumerate(polygon_text_list):
                polygon = get_polygon_from_text(polygon_text.text)
                information['geometry'] = polygon

            gs_features.append(GeoSeries(information))

    frame = GeoDataFrame(gs_features)
    open(outfile, 'wb').write(frame.to_json())


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('outfile', type=click.Path(writable=True))
def main(input_dir, outfile):
    citygml_to_geojson(input_dir, outfile)

if __name__ == '__main__':
    main()

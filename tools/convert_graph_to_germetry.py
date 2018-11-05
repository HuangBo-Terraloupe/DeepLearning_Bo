from shapely.geometry import MultiLineString
import os
import json
import numpy as np
import geopandas as gpd

def convert_graph_geometry(graph_file):
    '''convert graph file to multi-line-string
    Args:
        graph_file: path.../.graph

    Returns:
        multi-line-string
    '''
    data = []
    with open(graph_file, 'r') as fp:
        for line in fp:
            data.append([int(f) for f in line.split()])
    split_index = data.index([])
    lines = data[0:split_index]
    connect_infos = data[split_index + 1:][0::2]
    coords = []
    for connect_info in connect_infos:
         coords.append((tuple(lines[connect_info[0]]), tuple(lines[connect_info[1]])))
    multi_lines = MultiLineString(coords)
    return multi_lines



def convert_list_to_string(list_of_list):
    '''Convert a list of list to a list of string
    Args:
        list_of_list:

    Returns:
        list of string
    '''
    out_data_string = []
    for item in list_of_list:
        out_data_string.append(str(int(item[0])) + ' ' + str(int(item[1])))
    return out_data_string

def convert_geometry_to_graph(input_geojson, output_graph, convert_xy=False):
    '''Convert geojson tp graph file
    Args:
        input_geojson: input geojson file
        output_graph: output graph file path

    Returns:
            None
    '''
    # find the points
    points = []
    df = gpd.read_file(input_geojson)

    if convert_xy is False:
        for _, geo in enumerate(df['geometry']):
            for point in np.array(geo):
                if point.tolist() not in points:
                    points.append(point.tolist())

        # find the connection info
        connection_infos = []
        for _, geo in enumerate(df['geometry']):
            point_1 = points.index(np.array(geo)[0].tolist())
            point_2 = points.index(np.array(geo)[1].tolist())
            connection_infos.append([point_1, point_2])
            connection_infos.append([point_2, point_1])
    else:
        for _, geo in enumerate(df['geometry']):
            for point in np.array(geo):
                convert_point = [-point.tolist()[1], point.tolist()[0]]
                if convert_point not in points:
                    points.append(convert_point)

        # find the connection info
        connection_infos = []
        for _, geo in enumerate(df['geometry']):
            point_1 = [-np.array(geo)[0].tolist()[1], np.array(geo)[0].tolist()[0]]
            point_2 = [-np.array(geo)[1].tolist()[1], np.array(geo)[1].tolist()[0]]
            point_1_index = points.index(point_1)
            point_2_index = points.index(point_2)
            connection_infos.append([point_1_index, point_2_index])
            connection_infos.append([point_2_index, point_1_index])

    # write them into graph file
    points = convert_list_to_string(points)
    connection_infos = convert_list_to_string(connection_infos)
    out_data = points + [''] + connection_infos

    with open(output_graph, 'w') as f:
        for item in out_data:
            f.write("%s\n" % item)


def write_json(data, output_name, output_dir, geo_flag=True, indent=4):
    """
    Function to save (geo)data as *.geojson (needs to have .crs information!)

    Args:
        data: GeoDataFrame in memory
        output_name: desired output name
        output_dir: desired output folder for saving final output data
        geo_flag: boolean for saving *.geojson + CRS info
        indent: desired indent level for the output json

    Returns:
        *.geojson file in output directory based on EPSG of vector data

    """

    # if *.geojson is desired
    if geo_flag:

        # build crs header information to the header (easier I/O)
        epsg_out = int(data.crs['init'].replace('epsg:', ''))
        header = {
            'type': 'name',
            'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % epsg_out}
        }

        # add header to dictionary
        result = json.loads(data.to_json())
        result['crs'] = header

        # build file name
        file_ext = 'geojson'

    else:
        result = data

        # build file name
        file_ext = 'json'

    # save as *.geojson
    with open(os.path.join(output_dir, '%s.%s' % (output_name, file_ext)), 'w') as fp:
        json.dump(result, fp, indent=indent, sort_keys=True)


def get_extrapolated_line(p1, p2):
    """ Creates a line extrapolated in p1->p2 direction. """

    extrapolation_ratio = 1.5
    a = p1
    b = (p1[0]+extrapolation_ratio*(p2[0]-p1[0]), p1[1]+extrapolation_ratio*(p2[1]-p1[1]))
    return [a, b]

def calculate_angle(prev_point, curr_point, next_point):
    a = np.array(prev_point[:2])
    b = np.array(curr_point[:2])
    c = np.array(next_point[:2])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def remove_diverge_center_line(input_geojson, output_name, output_dir, get_point_percentage, length_threshold=None,
                               angle_threshold=None):
    '''remove the diverge center line from geojson.
    Args:
        input_geojson: input geojson file
        output_name: the name of geojson, exsample: 1.geojson
        output_dir: the output folder to save the geojson
        length_threshold: the threshold to keep the lines, if the line smaller than the number, it will be keeped.
        angle_threshold: the threshold of angle to keep the lines, if the anger between diverge line and main line is
        bigger than the threshold, the line will be keeped
        get_point_percentage: percentage to move from the main line, to find a point to calculate the angle

    Returns:

    '''

    df = gpd.read_file(input_geojson)

    points = []
    connection_infos_1 = []
    connection_infos_2 = []
    two_side_free_geo_index = []

    spatial_index = df.sindex

    for row_index, row in df.iterrows():
        geo = row['geometry']
        for point in np.array(geo):
            if point.tolist() not in points:
                points.append(point.tolist())
        point_1 = points.index(np.array(geo)[0].tolist())
        point_2 = points.index(np.array(geo)[1].tolist())
        connection_infos_1.append([point_2, point_1])
        connection_infos_2.append(point_1)
        connection_infos_2.append(point_2)

        possible_matches_index = list(spatial_index.intersection(geo.bounds))
        possible_matches = df.iloc[possible_matches_index]

        if list(possible_matches['geometry'].intersects(geo)).count(True) == 1:
            two_side_free_geo_index.append(row_index)
    print('the number of lines which has two free point:', len(two_side_free_geo_index))

    idx_only_show_once = []
    for connect_idx_1 in set(connection_infos_2):
        if connection_infos_2.count(connect_idx_1) == 1:
            idx_only_show_once.append(connect_idx_1)

    idx_remove_row = []
    for connect_idx_2, connection in enumerate(connection_infos_1):
        if len(set(idx_only_show_once).intersection(connection)) > 0 and connect_idx_2 not in two_side_free_geo_index:
            idx_remove_row.append(connect_idx_2)
    print('the number of lines which has one free point:', len(idx_remove_row))

    # remove the lines if it is too short
    if length_threshold is not None:
        longer_line_list = []
        for m in idx_remove_row:
            if float(df['geometry'][m].length) > length_threshold:
                longer_line_list.append(m)
        idx_remove_row = list(set(idx_remove_row) - set(longer_line_list))
        print('the number of lines to be removed after keeping the lines longer than threshold:', len(idx_remove_row))

    if angle_threshold is not None:
        # keep the the line if the angle is bigger than the threshold, not diverges too much from the main line
        bigger_angle_line_list = []
        for idx in idx_remove_row:
            intersection_list = np.array(df['geometry'].intersects(df['geometry'][idx]))
            intersections = np.where(intersection_list == True)[0]
            intersections = list(intersections)
            intersections.remove(idx)

            if len(intersections) == 1:
                main_line = df['geometry'][int(intersections[0])]
            elif len(intersections) == 0:
                continue
            elif len(intersections) > 1:
                main_line = df['geometry'][df['geometry'][intersections].length.idxmax()]
            else:
                raise ValueError('the number of intersection geometry is wrong.')

            curr_point = np.array(main_line.intersection(df['geometry'][idx])).tolist()
            prev_geo_points = np.array(df['geometry'][idx]).tolist()
            prev_point = prev_geo_points[1 - prev_geo_points.index(curr_point)]
            next_point = np.array(main_line.interpolate(get_point_percentage, normalized=True)).tolist()
            angle = calculate_angle(prev_point, curr_point, next_point)

            if angle > angle_threshold:
                bigger_angle_line_list.append(idx)
        idx_remove_row = list(set(idx_remove_row) - set(bigger_angle_line_list))
        print('the number of lines finally to be removed:', len(idx_remove_row))

    # save the geometry after remove the diverge center line
    df_save = df.drop(idx_remove_row)
    write_json(df_save, output_name, output_dir)

if __name__ == '__main__':
    input_geojson = '/home/terraloupe/18JUN30174106-S3DS-058526470080_01_P001.geojson'
    #input_geojson = '/home/terraloupe/Houston_lower_100kx250k_000000_020000.geojson'
    output_name = 'bo_output_v3.geojson'
    output_dir = '/home/terraloupe'
    get_point_percentage = 0.05
    length_threshold = 20
    angle_threshold = 150

    remove_diverge_center_line(input_geojson, output_name, output_dir, get_point_percentage, length_threshold,
                               angle_threshold)
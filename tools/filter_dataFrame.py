def run_filter(input_file, epsg_out, output_name, output_dir):
    df = gpd.read_file(input_file)
    lines_geo_series = []
    for ind, serie in df.iterrows():
        geom = serie.geometry
        if isinstance(geom, MultiLineString):
            for line in geom:
                lines_geo_series.append(gpd.GeoSeries({'geometry': line, 'category': serie.road_type}))
        elif isinstance(geom, LineString):
            lines_geo_series.append(gpd.GeoSeries({'geometry': geom, 'category': serie.road_type}))
        else:
            print("Null geometry detected")

    df_new = GeoDataFrame(lines_geo_series)
    df_new.crs = {'init': 'epsg:%d' % epsg_out}
    write_json(df_new, output_name=output_name,
               output_dir,
               geo_flag=True,
               indent=4)
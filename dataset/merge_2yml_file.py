import click
import yaml

@click.command()
@click.argument("yml_files_one", type=click.Path(exists=True, dir_okay=False))
@click.argument("yml_files_two", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_yml", type=click.Path(exists=False, dir_okay=False))
def main(yml_files_one, yml_files_two, output_yml):

    with open(yml_files_one, 'rb') as fp:
        data1 = yaml.load(fp.read())

    with open(yml_files_two, 'rb') as fp:
        data2 = yaml.load(fp.read())

    train = {'images': data1['training']['images'] + data2['training']['images'],
             'labels': data1['training']['labels'] + data2['training']['labels']}
    val = {'images': data1['validation']['images'] + data2['validation']['images'],
             'labels': data1['validation']['labels'] + data2['validation']['labels']}
    test = {'images': data1['testing']['images'] + data2['testing']['images'],
             'labels': data1['testing']['labels'] + data2['testing']['labels']}

    import pdb
    pdb.set_trace()

    dataset = {"name": data1['name'],
               "prefix": data1['prefix'],
               "data": data1['data'],
               "ground_truth": data1['ground_truth'],
               'training': train,
               'validation': val,
               'testing': test
               }



    f = open(output_yml, 'w')
    yaml.dump(dataset, f, default_flow_style=False)


if __name__ == '__main__':
    main()
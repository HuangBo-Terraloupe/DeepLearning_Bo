import click
import yaml
import numpy as np

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

    try:
        index_array = np.random.permutation(len(train['images']))
        train['images'] = list(np.array(train['images'])[index_array])
        train['labels'] = list(np.array(train['labels'])[index_array])
    except:
        print('training set not shuffeled ...')
        pass

    try:
        index_array = np.random.permutation(len(val['images']))
        val['images'] = list(np.array(val['images'])[index_array])
        val['labels'] = list(np.array(val['labels'])[index_array])
    except:
        print('validation set not shuffeled ...')
        pass

    try:
        index_array = np.random.permutation(len(test['images']))
        test['images'] = list(np.array(test['images'])[index_array])
        test['labels'] = list(np.array(test['labels'])[index_array])
    except:
        print('testing set not shuffeled ...')
        pass

    import pdb
    pdb.set_trace()

    dataset = {"name": 'Merge',
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
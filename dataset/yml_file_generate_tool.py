import os
import yaml

from glob import glob
from random import shuffle

import click


def generate_yml(experiment_name, image_type, image_channel, source_bitdepth, class_names, gt_type, prefix,
                 images_dir_name, masks_dir_name, image_ext, mask_ext, outputfile_name, split_ratio):
    '''
    Args:
        experiment_name:
        image_type:
        image_channel:
        source_bitdepth:
        class_names:
        gt_type:
        prefix:
        images_dir_name:
        masks_dir_name:
        image_ext:
        mask_ext:
        outputfile_name:
        split_ratio:

    Returns:

    '''
    data = {"type": image_type, "channels": image_channel, "source_bitdepth": source_bitdepth}
    classes = [{"index": i, "name": cls} for i, cls in enumerate(class_names.split(','))]
    ground_truth = {"type": gt_type, "classes": classes}

    # Get image and label lists
    images_dir = os.path.join(prefix, images_dir_name)
    masks_dir = os.path.join(prefix, masks_dir_name)
    image_paths = []
    masks_paths = []

    image_ids = glob(images_dir + '/*' + image_ext)
    mask_ids = glob(masks_dir + '/*' + mask_ext)

    image_ids_names = [os.path.split(f)[1].split('.')[0] for f in image_ids]
    mask_ids_names = [os.path.split(f)[1].split('.')[0] for f in mask_ids]

    common_names = list(set(image_ids_names).intersection(mask_ids_names))
    shuffle(common_names)

    for name in common_names:
        image_path = os.path.join(images_dir, name + image_ext)
        image_paths.append(image_path)

        masks_path = os.path.join(masks_dir, name + mask_ext)
        masks_paths.append(masks_path)

    total_images = len(image_paths)
    split_ratio = split_ratio.split(',')
    n_train = int(total_images * split_ratio[0])
    n_val = int(total_images * split_ratio[1])
    n_test = int(total_images * split_ratio[2])

    train_list = image_paths[0:n_train]
    val_list = image_paths[n_train: n_train + n_val]
    test_list = image_paths[n_train + n_val: n_train + n_val + n_test]

    mask_train_list = masks_paths[0:n_train]
    mask_val_list = masks_paths[n_train: n_train + n_val]
    mask_test_list = masks_paths[n_train + n_val: n_train + n_val + n_test]

    train = {'images': train_list, 'labels': mask_train_list}
    val = {'images': val_list, 'labels': mask_val_list}
    test = {'images': test_list, 'labels': mask_test_list}

    dataset = {"name": experiment_name,
               "prefix": prefix,
               "data": data,
               "ground_truth": ground_truth,
               'training': train,
               'validation': val,
               'testing': test
               }

    f = open(outputfile_name, 'w')
    yaml.dump(dataset, f, default_flow_style=False)


@click.command()
@click.argument('experiment_name', type=str)
@click.argument("prefix", type=click.Path(exists=False, dir_okay=True))
@click.argument("images_dir_name", type=click.Path(exists=False, dir_okay=True))
@click.argument("masks_dir_name", type=click.Path(exists=False, dir_okay=True))
@click.argument("outputfile_name", type=click.Path(exists=True, dir_okay=True))

@click.option('--image_type', type=str, default='ortho')
@click.option('--image_channel', type=str, default='rgb')
@click.option('--source_bitdepth', type=int, default=8)
@click.option('--class_names', type=str, default='', help="a string of class names, exsample: 'cat, dog' ")
@click.option("--gt_type", type=click.Choice(['semseg', 'categorical', 'binary', 'sparse', None]))
@click.option("--image_ext", type=str, default='.png')
@click.option("--mask_ext", type=str, default='.png')
@click.option("--split_ratio", type=str, default='0.9, 0.1, 0.0')


def main(experiment_name, image_type, image_channel, source_bitdepth, class_names, gt_type, prefix,
                 images_dir_name, masks_dir_name, image_ext, mask_ext, outputfile_name, split_ratio):

    generate_yml(experiment_name, image_type, image_channel, source_bitdepth, class_names, gt_type, prefix,
                     images_dir_name, masks_dir_name, image_ext, mask_ext, outputfile_name, split_ratio)

if __name__ == '__main__':
    main()

import json
import csv
import os
import yaml
import click


def covnert(data_file, save_dir):
    '''convert the json annotation to csv file for the training of retinanet
    :param data_file: yaml file
    :param save_dir: save dir to save the csv
    :return:
    '''

    with open(data_file, 'r') as fp:
        spec = yaml.load(fp)
    annotation_training = spec['training']['labels']
    annotation_val = spec['validation']['labels']
    annotation_test = spec['testing']['labels']

    # save the class_mapping into csv
    classes = spec['ground_truth']['classes']
    with open(os.path.join(save_dir, 'class_mapping.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for _, cls in enumerate(classes):
            writer.writerow([cls['name'], cls['index']])


    # training
    print 'operating training data'
    if len(annotation_training) == 0:
        pass
    elif len(annotation_training) > 0:
        with open(os.path.join(save_dir, 'training.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for annotation in annotation_training:
                with open(spec['prefix'] + annotation, 'r') as fp:
                    data = json.load(fp)
                    for bbox in data['bboxes']:
                        filepath = spec['prefix'] + 'images/' + data['img_name']
                        x1 = bbox['x1']
                        x2 = bbox['x2']
                        y1 = bbox['y1']
                        y2 = bbox['y2']
                        class_name = bbox['category']
                        spamwriter.writerow([filepath, x1, y1, x2, y2, class_name])

    # validation
    print 'operating validation data'
    if len(annotation_val) == 0:
        pass
    elif len(annotation_val) > 0:
        with open(os.path.join(save_dir, 'validation.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for annotation in annotation_val:
                with open(spec['prefix'] + annotation, 'r') as fp:
                    data = json.load(fp)
                    for bbox in data['bboxes']:
                        filepath = spec['prefix'] + 'images/' + data['img_name']
                        x1 = bbox['x1']
                        x2 = bbox['x2']
                        y1 = bbox['y1']
                        y2 = bbox['y2']
                        class_name = bbox['category']
                        spamwriter.writerow([filepath, x1, y1, x2, y2, class_name])

    # test
    print 'operating test data'
    if len(annotation_test) == 0:
        pass
    elif len(annotation_test) > 0:
        with open(os.path.join(save_dir, 'test.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for annotation in annotation_test:
                with open(spec['prefix'] + annotation, 'r') as fp:
                    data = json.load(fp)
                    for bbox in data['bboxes']:
                        filepath = spec['prefix'] + 'images/' + data['img_name']
                        x1 = bbox['x1']
                        x2 = bbox['x2']
                        y1 = bbox['y1']
                        y2 = bbox['y2']
                        class_name = bbox['category']
                        spamwriter.writerow([filepath, x1, y1, x2, y2, class_name])


@click.command()
@click.argument('data_file')
@click.argument('save_dir')

def main(data_file, save_dir):
    covnert(data_file, save_dir)


if __name__ == '__main__':
    main()


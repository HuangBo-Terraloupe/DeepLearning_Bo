import os
import csv
import yaml
import pandas as pd
import numpy as np


def calculate_overlaps(df, ratio):
    total_pixels = df['background'][0] + df['roads'][0]
    overlaps = df['roads'] / total_pixels
    nb_negative = len(overlaps[overlaps == 0])
    nb_median = len(overlaps[overlaps < ratio]) - len(overlaps[overlaps == 0])
    nb_positive = len(overlaps[overlaps > ratio])

    print('positives:', nb_positive)
    print('median:', nb_median)
    print('negative:', nb_negative)
    print('total number:', nb_positive + nb_median + nb_negative)

    return nb_positive, nb_median, nb_negative


def random_sample_trainingdata(image_list, mask_list, n_samples):
    validation_index = np.random.choice(len(image_list), n_samples, replace=False)
    validation_images = np.array(image_list)[validation_index].tolist()
    validation_labels = np.array(mask_list)[validation_index].tolist()

    train_images = list(set(image_list) ^ set(validation_images))
    train_labels = list(set(mask_list) ^ set(validation_labels))

    train_images.sort(), train_labels.sort()

    indexes = np.random.permutation(np.arange(len(train_images)))
    train_images = np.array(train_images)[indexes].tolist()
    train_labels = np.array(train_labels)[indexes].tolist()
    return train_images, train_labels, validation_images, validation_labels

houston = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/houstion.csv')
san_francisco = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/san_francisco.csv')
texas1 = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/texas1.csv')
texas2 = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/texas2.csv')
south = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/here_osm_roads%2Fpilot_imagery_south_america%2Fmask_stats.csv')
north = pd.DataFrame.from_csv('/home/terraloupe/Dataset/50_30_data/here_osm_roads%2Fpilot_imagery_north_america%2Fmask_stats.csv')

rest = [houston, san_francisco, texas1, texas2]
poc = [south, north]
rest_result = pd.concat(rest)
poc_result = pd.concat(poc)

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/san_francisco.yml'
with open(yaml_file, 'rb') as fp:
    san_francisco = yaml.load(fp.read())

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/houston.yml'
with open(yaml_file, 'rb') as fp:
    houston = yaml.load(fp.read())

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/texas1.yml'
with open(yaml_file, 'rb') as fp:
    texas1 = yaml.load(fp.read())

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/texas2.yml'
with open(yaml_file, 'rb') as fp:
    texas2 = yaml.load(fp.read())

rest_total_images = san_francisco['training']['images'] + houston['training']['images'] + texas1['training']['images'] + \
                    texas2['training']['images']
rest_total_masks = san_francisco['training']['labels'] + houston['training']['labels'] + texas1['training']['labels'] + \
                   texas2['training']['labels']

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/north.yml'
with open(yaml_file, 'rb') as fp:
    north_yml = yaml.load(fp.read())

yaml_file = '/home/terraloupe/Dataset/50_30_data/ymls/south.yml'
with open(yaml_file, 'rb') as fp:
    south_yml = yaml.load(fp.read())

poc_total_images = north_yml['training']['images'] + south_yml['training']['images']
poc_total_masks = north_yml['training']['labels'] + south_yml['training']['labels']

rest_train_images, rest_train_labels, rest_val_images, rest_val_labels = random_sample_trainingdata(rest_total_images, rest_total_masks, n_samples=25000)
poc_train_images, poc_train_labels, poc_val_images, poc_val_labels = random_sample_trainingdata(poc_total_images, poc_total_masks, n_samples=329)

rest_train_images_names = [os.path.split(f)[1].split('.')[0] for f in rest_train_images]
rest_train_df = rest_result.loc[rest_train_images_names]

poc_train_images_names = [os.path.split(f)[1].split('.')[0] for f in poc_train_images]
poc_train_df = poc_result.loc[poc_train_images_names]

rest_train_df['sum'] = rest_train_df['background'] + rest_train_df['roads']
rest_train_df['ratio'] = rest_train_df['roads'] / rest_train_df['sum']
poc_train_df['sum'] = poc_train_df['background'] + poc_train_df['roads']
poc_train_df['ratio'] = poc_train_df['roads'] / poc_train_df['sum']

n_res_poc_positive = len(rest_train_df['ratio'][rest_train_df['ratio'] >=0.1])
n_res_poc_median = len(rest_train_df['ratio'][rest_train_df['ratio'] < 0.1]) - len(rest_train_df['ratio'][rest_train_df['ratio'] == 0])
n_res_poc_negative = len(rest_train_df['ratio'][rest_train_df['ratio'] ==0])

n_poc_positive = len(poc_train_df['ratio'][poc_train_df['ratio'] >= 0.1])
n_poc_median = len(poc_train_df['ratio'][poc_train_df['ratio'] < 0.1]) - len(poc_train_df['ratio'][poc_train_df['ratio'] == 0])
n_poc_negative = len(poc_train_df['ratio'][poc_train_df['ratio'] ==0])

print('rest', n_res_poc_positive, n_res_poc_median, n_res_poc_negative, 'poc', n_poc_positive, n_poc_median, n_poc_negative)

rest_positive_prob = 0.9933 * (1.0 / 3) *  (1.0 / n_res_poc_positive)
rest_median_prob = 0.9933 * (1.0 / 3) *  (1.0 / n_res_poc_median)
rest_negative_prob = 0.9933 * (1.0 / 3) *  (1.0 / n_res_poc_negative)
print(rest_positive_prob, rest_median_prob, rest_negative_prob)

poc_positive_prob = 0.0067 * (1.0 / 3) *  (1.0 / n_poc_positive)
poc_median_prob = 0.0067 * (1.0 / 3) *  (1.0 / n_poc_median)
poc_negative_prob = 0.0067 * (1.0 / 3) *  (1.0 / n_poc_negative)
print(poc_positive_prob, poc_median_prob, poc_negative_prob)

def apply_probability(row):
    if row['ratio'] == 0.0:
        return poc_negative_prob
    elif row['ratio'] >= 0.1:
        return poc_positive_prob
    else:
        return poc_median_prob

poc_train_df["probability"] = poc_train_df.apply(apply_probability, axis=1)

def calc_prob(row):
    if row['ratio'] == 0.0:
        return rest_negative_prob
    elif row['ratio'] >= 0.1:
        return rest_positive_prob
    else:
        return rest_median_prob

rest_train_df["probability"] = rest_train_df.apply(calc_prob, axis=1)

rest_train_probs = rest_train_df['probability'].tolist()
poc_train_probs = poc_train_df['probability'].tolist()

train = {'images': rest_train_images + poc_train_images, 'labels': rest_train_labels + poc_train_labels, 'probability':rest_train_probs + poc_train_probs}
val = {'images': rest_val_images + poc_val_images, 'labels': rest_val_labels + poc_val_labels, 'probability':[]}
test = {'images': [], 'labels': [], 'probability':[]}

dataset = {"name": 'Here training Data',
           "prefix": north_yml['prefix'],
           "data": north_yml['data'],
           "ground_truth": north_yml['ground_truth'],
           'training': train,
           'validation': val,
           'testing': test
           }

f = open('/home/terraloupe/Dataset/50_30_data/ymls/5030cm_merge_sampling_YMLinput_v1.yml', 'w')
yaml.dump(dataset, f, default_flow_style=False)


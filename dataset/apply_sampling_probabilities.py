import os
import cv2
import yaml
import numpy as np


def run(yml_file, yml_save_path, category_diff_ratio=None):

    with open(yml_file, 'rb') as fp:
        spec = yaml.load(fp.read())

    train_prob_list = []
    validation_prob_list = []
    testing_prob_list = []

    train_masks = spec["training"]["labels"]
    if len(train_masks) > 0:
        print('calculating sampling probabilities for training set ...')

        static_dic = {'negative': 0, 'positive': 0, 'median': 0}
        for i, file in enumerate(train_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                static_dic['negative'] = static_dic['negative'] + 1
            elif (float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1])) > category_diff_ratio:
                print((float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1])))
                print(category_diff_ratio)
                static_dic['positive'] = static_dic['positive'] + 1
            else:
                static_dic['median'] = static_dic['median'] + 1

        try:
            prob_negative = 1.0 / (static_dic['negative'] * 3)
        except:
            prob_negative = 0

        try:
            prob_positive = 1.0 / (static_dic['positive'] * 3)
        except:
            prob_positive = 0

        try:
            prob_median = 1.0 / (static_dic['median'] * 3)
        except:
            prob_median = 0


        print('Training statistic Info:')
        print(static_dic)
        print('positive, median, negative', prob_positive, prob_median, prob_negative)

    
        for i, file in enumerate(train_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                train_prob_list.append(prob_negative)
            elif (float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1]))  > category_diff_ratio:
                train_prob_list.append(prob_positive)
            else:
                train_prob_list.append(prob_median)

    validation_masks = spec["validation"]["labels"]
    if len(validation_masks) > 0:
        print('calculating sampling probabilities for validation set ...')
        static_dic = {'negative': 0, 'positive': 0, 'median': 0}
        for i, file in enumerate(validation_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                static_dic['negative'] = static_dic['negative'] + 1
            elif (float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1]))  > category_diff_ratio:
                static_dic['positive'] = static_dic['positive'] + 1
            else:
                static_dic['median'] = static_dic['median'] + 1

        try:
            prob_negative = 1.0 / (static_dic['negative'] * 3)
        except:
            prob_negative = 0

        try:
            prob_positive = 1.0 / (static_dic['positive'] * 3)
        except:
            prob_positive = 0

        try:
            prob_median = 1.0 / (static_dic['median'] * 3)
        except:
            prob_median = 0

        print('Validation statistic Info:')
        print(static_dic)
        print('positive, median, negative', prob_positive, prob_median, prob_negative)



        for i, file in enumerate(validation_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                validation_prob_list.append(prob_negative)
            elif (float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1]))  > category_diff_ratio:
                validation_prob_list.append(prob_positive)
            else:
                validation_prob_list.append(prob_median)


    testing_masks = spec["testing"]["labels"]
    if len(testing_masks) > 0:
        print('calculating sampling probabilities for validation set ...')
        static_dic = {'negative': 0, 'positive': 0, 'median': 0}
        for i, file in enumerate(testing_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                static_dic['negative'] = static_dic['negative'] + 1
            elif float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1]) > category_diff_ratio:
                static_dic['positive'] = static_dic['positive'] + 1
            else:
                static_dic['median'] = static_dic['median'] + 1

        try:
            prob_negative = 1.0 / (static_dic['negative'] * 3)
        except:
            prob_negative = 0

        try:
            prob_positive = 1.0 / (static_dic['positive'] * 3)
        except:
            prob_positive = 0

        try:
            prob_median = 1.0 / (static_dic['median'] * 3)
        except:
            prob_median = 0

        print('Testing statistic Info:')
        print(static_dic)
        print('positive, median, negative', prob_positive, prob_median, prob_negative)



        for i, file in enumerate(testing_masks):
            mask_file = os.path.join(spec['prefix'], file)
            mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
            if np.all(mask_file==0):
                testing_prob_list.append(prob_negative)
            elif (float(np.count_nonzero(mask_file)) / (mask_file.shape[0] * mask_file.shape[1]))  > category_diff_ratio:
                testing_prob_list.append(prob_positive)
            else:
                testing_prob_list.append(prob_median)

    print('probabilities are done... now save them into yml file ...')
    train = {'images': spec["training"]["images"], 'labels': spec["training"]["labels"], 'probability': train_prob_list}
    val = {'images': spec["validation"]["images"], 'labels': spec["validation"]["labels"], 'probability': validation_prob_list}
    test = {'images': spec["testing"]["images"], 'labels': spec["testing"]["labels"], 'probability': testing_prob_list}

    dataset = {"name": spec['name'],
               "prefix": spec['prefix'],
               "data": spec['data'],
               "ground_truth": spec['ground_truth'],
               'training': train,
               'validation': val,
               'testing': test
               }

    f = open(os.path.join(spec['prefix'], yml_save_path), 'w')
    yaml.dump(dataset, f, default_flow_style=False)


if __name__ == '__main__':
    yml_file = '/home/terraloupe/san_francisco.yml'
    yml_save_path = '/home/terraloupe/sampled_san_francisco.yml'
    run(yml_file, yml_save_path, category_diff_ratio=0.1)

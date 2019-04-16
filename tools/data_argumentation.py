from keras.preprocessing.image import ImageDataGenerator


def data_argumentation(img_width, img_height, total_num_samples, batch_size, image_data_dir, mask_data_dir,
                       save_to_dir_image, save_to_dir_mask):
    '''Do data argumentation using keras generator, it will generate both images and masks which are data-argumented.
    Args:
        img_width: pixel value
        img_height: pixel value
        total_num_samples:
        batch_size:
        image_data_dir:
        mask_data_dir:
        save_to_dir_image:
        save_to_dir_mask:

    Returns:

    '''


    operation_epoch = total_num_samples // batch_size

    data_gen_args = dict(brightness_range=[1.0, 1.5])
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        directory=image_data_dir,
        target_size=(img_width, img_height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        save_to_dir=save_to_dir_image,
        save_prefix='brightness',
        save_format='jpeg',
        seed=0
    )

    mask_generator = mask_datagen.flow_from_directory(
        directory=mask_data_dir,
        target_size=(img_width, img_height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        save_to_dir=save_to_dir_mask,
        save_prefix='brightness',
        save_format='jpeg',
        seed=0
    )

    print('operation image set ....')
    for m in range(operation_epoch):
        print(m)
        if m ==100:
            break
        index = 0
        for batch in image_generator:
            if index == 1:
                break
            index = index + 1

    print('operation mask set ....')
    for n in range(operation_epoch):
        print(n)
        if m ==100:
            break
        index = 0
        for batch in mask_generator:
            if index == 1:
                break
            index = index + 1

if __name__ == '__main__':

    batch_size = 32
    total_num_samples = 36004
    img_width, img_height = 1024, 1024
    image_data_dir = '/home/terraloupe/Dataset/germany_road_area/ger_alpha_road_area/train/lane_marking/images'
    mask_data_dir = '/home/terraloupe/Dataset/germany_road_area/ger_alpha_road_area/train/lane_marking/masks'
    save_to_dir_image = '/home/terraloupe/Dataset/germany_road_area/ger_alpha_road_area/data_argumentation/brightness/images'
    save_to_dir_mask = '/home/terraloupe/Dataset/germany_road_area/ger_alpha_road_area/data_argumentation/brightness/masks'

    data_argumentation(img_width, img_height, total_num_samples, batch_size, image_data_dir, mask_data_dir,
                       save_to_dir_image, save_to_dir_mask)
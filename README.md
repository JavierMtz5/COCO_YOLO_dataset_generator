# COCO-YOLO Dataset Generator v0.1.0

![COCO-YOLOv8 converter](assets/coco_yolo.png)

## Dataset structure

The script expects to have the COCO format dataset at the same level as COCO_dataset_extractor, following this structure:

- dataset_directory_name
    - annotations
        - instances_train2017.json
        - instances_val2017.json
    - images
        - train2017
            - image_00001.jpg
            - image_00002.jpg
            - ...
        - val2017
            - image_00100.jpg
            - image_00101.jpg
            - ...
        - test2017
            - image_00201.jpg
            - image_00202.jpg
            - ...

Image filenames do not have to be in order nor named in the same way, and the main directory (*dataset_directory_name* in the diagram above) containing the dataset can be named by the user. The remaining directories must be named in the same way.

It is not mandatory to have the original COCO dataset, but any other COCO format dataset must follow this structure so that the script can correctly convert the data.

## Arguments

- **dataset_dir**: Path to the directory where COCO dataset is located.
- **output_dir**: Name of the directory where the new dataset will be generated. Defaults to *new_dataset*.
- **target_classes**: Array of strings, where each string is the name of the class whose images that must be extracted from the original COCO dataset. If not specified, all classes are extracted from the original dataset.
- **background_percentage**: Only applies if not all classes are being extracted from COCO dataset. The new dataset will include *background_percentage*% more images, which will be background and will not contain any of the target classes. Defaults to 0.0.
- **create_single_class**: Boolean indicating whether to join all the selected classes into a single class. Defaults to False.
- **single_class_name**: Only applies if create_single_class param is set to True. Name of the single class to be generated. Defaults to *new_class*.
- **test_num_images**: Number of test images from the original COCO dataset to include in the new dataset. Defaults to None, which means that all of the original test images will be included.
- **test_only_target_classes** (**AVAILABLE IN FUTURE RELEASE**): Boolean indicating whether to only include images which contain the target classes or any image. Defaults to False.
- **convert_to_yolo** (**AVAILABLE IN FUTURE RELEASE**): Boolean indicating whether to convert the annotations to YOLOv8 or not. Defaults to True.

## Usage

### Dependencies

Install the required dependencies by running the following command from the *COCO_YOLO_dataset_generator* directory:

```bash
$ pip install -r requirements. txt
```

### Execution

- The usage of the script includes different possibilities and features. To extract the desired dataset, many of the parameters above must be correctly defined.

- The *dataset_dir* parameter is the only mandatory parameter. It must be the name of the directory where the original dataset with COCO format is located.

- For the remaining parameters, to know which ones you must set, read each of the questions below and set the corresponding parameter if it applies:

#### Do I want a custom directory name for the new dataset?

If you want to give a name to the directory which will contain the new dataset, set the *output_dir* parameter with the desired name. Do not use spaces.

#### Do I want to extract only images containing objects that belong to certain classes, or all the images of the original dataset?

If you want to extract only images which contain objects from certain classes, you must set the *target_classes* parameter with an array of strings, which are the names of the classes you want to keep. For example, if instead of loading 118K images with all different objects, you only want to load images which contain dogs and cats, then *target_classes* parameter shuould be set to ```--target_classes cat dog```.

If you want to extract all the original images, do not set a value for the *target_classes* parameter.

#### Do you want to include background images in the new dataset?

If you gave a value to the *target_classes* parameter, the new dataset will now contain only images with the target classes, which can lead to an increase in false positives. To solve this problem, a common solution is to add images which do not contain any of the target classes to the dataset. Parameter *background_percentange* sets how many background images will be added to the new dataset. More specifically, the number of background images to be added is equal to the length of the new dataset (without background images) multiplied by the *background_percentage* parameter, which must be a value between 0 and 1.

If the length of the new dataset is 1000, and I set the *background_percentage* parameter to 0.2, then I will add 200 background images to the new dataset. If the *background_percentage* parameter is not set, then no background images will be added.

#### How many test images from the original dataset do you want to include in the new test dataset?

If, instead of 100K test images you only want to extract 500 test images, then *test_num_images* parameter must be set to 500. If not set, all original test images will be included in the new test set.

#### Do you want to extract only the test images which contain the target classes?

If you want to extract only images containing objects that belong to the target classes, then *test_exclude_classes* must be set to False. If *test_exclude_classes* parameter is not set, then all original test images, independently of the objects they contain, will be extracted.

#### Do you want to join all the classes of the new dataset into one single class?

The original COCO dataset contains images with cats and dogs. These two classes can be converted to a single class: *animal*. If you want to change all the annotations from the original x-classes dataset to a new 1-class dataset, parameter *create_single_class* must be set to True.

If you want to specify a name for the new class, then you must set *single_class_name* to the name you want.

#### Do you want to convert from the original COCO format dataset to YOLOv8 format, or do you just want to extract certain classes but keep the COCO format?

If you want the new dataset to be YOLOv8 format, then you must set *convert_to_yolo* parameter to True (by default it is set to True). If you want the new dataset to keep the COCO format, then you must set the *convert_to_yolo* parameter to False.

### Usage example

1. ##### Convert COCO format dataset to YOLOv8 format, extracting all the original images and annotations. Extract all original test images.

```bash
$ python3 coco_to_yolo_extractor.py coco_dataset_directory --convert_to_yolo true --output_dir new_dataset_directory
```

2. ##### Convert COCO format dataset to YOLOv8 format, extracting only images containing 'dog' and 'cat' classes from the original dataset. Extract all original test images.

```bash
$ python3 coco_to_yolo_extractor.py coco_dataset_directory --convert_to_yolo true --target_classes dog cat --output_dir new_dataset_directory
```

3. ##### Convert COCO format dataset to YOLOv8 format, first extracting only images containing 'dog' and 'cat' classes, and remapping all 'dog' and 'cat' annotations to a single class 'animals'. Extract all original test images.

```bash
$ python3 coco_to_yolo_extractor.py coco_dataset_directory --convert_to_yolo true --target_classes dog cat --create_single_class true --single_class_name animals --output_dir new_dataset_directory
```

4. ##### Convert COCO format dataset to YOLOv8 format, first extracting only images containing 'dog' and 'cat' classes, and remapping all 'dog' and 'cat' annotations to a single class 'animals'. Add 20% of background images (images which do not contain any of the target classes) to the new dataset. Extract all original test images.

```bash
$ python3 coco_to_yolo_extractor.py coco_dataset_directory --convert_to_yolo true --target_classes dog cat --background_percentage 0.2 --create_single_class true --single_class_name animals --output_dir new_dataset_directory
```

5. ##### Convert COCO format dataset to YOLOv8 format, first extracting only images containing 'dog' and 'cat' classes, and remapping all 'dog' and 'cat' annotations to a single class 'animals'. Add 20% of background images (images which do not contain any of the target classes) to the new dataset. Only 1000 images from the original test set will be extracted.

```bash
$ python3 coco_to_yolo_extractor.py coco_dataset_directory --convert_to_yolo true --target_classes dog cat --background_percentage 0.2 --create_single_class true --single_class_name animals --output_dir new_dataset_directory --test_num_images 1000
```

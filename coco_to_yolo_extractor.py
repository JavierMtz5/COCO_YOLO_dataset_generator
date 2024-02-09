import os
import argparse
import random
from typing import List, Optional, Union, Dict, Tuple, Set
import json
from tqdm import tqdm
import shutil


class COCOConverter:

    def __init__(self):
        
        # Read arguments
        self.parse_arguments()

        # Create output directories
        for dataset_type in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.output_dir, dataset_type, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, dataset_type, 'labels'), exist_ok=True)

    def initialize_conversion(self) -> None:
        """Processes all sets of data and generates the new modified dataset"""

        # Process each dataset
        self.process_dataset('train',
                             self.coco_annotation_train,
                             self.coco_image_dir_train,
                             self.single_class_name if self.create_single_class else None)
        self.process_dataset('valid',
                             self.coco_annotation_val,
                             self.coco_image_dir_val,
                             self.single_class_name if self.create_single_class else None)

        self.process_dataset('test',
                             self.coco_annotation_test,
                             self.coco_image_dir_test)

    def process_dataset(self,
                        dataset_type: str,
                        coco_annotation_file: str,
                        coco_image_dir: str,
                        single_class_name: Optional[str] = None) -> None:
        """
        Processes the new dataset in different ways, depending if the data is train/val or test.
        Args:
            dataset_type (str): train/valid/test.
            coco_annotation_file (str): Path to the annotations of the original COCO dataset.
            coco_image_dir (str): Path to the directory which contains the images of the original COCO dataset.
            single_class_name (Optional[str]): Name of the single class to be generated, if specified by the create_single_class parameter.
        """

        print(f'\nProcessing the {dataset_type} data...')
        # Process Test data
        if dataset_type == 'test':
            self.process_test_data()

        # If COCO annotations exist, process Train or Validation data
        elif os.path.exists(coco_annotation_file):
            self.process_train_val_data(dataset_type, coco_annotation_file, coco_image_dir, single_class_name)

        else:
            print(f'Could not find a COCO-format annotation file in {coco_annotation_file}')

    def process_test_data(self) -> None:
        """
        Processes the test data.
        1. If only images containing target classes must be extracted, check if the annotation file exists.
            1.1. If annotation file exists, get the filenames of the images that contain target classes. Otherwise, extract all the available images.
            1.2. If only a limited number of images are to be extracted, sample a random batch of test images from the filenames extracted in step 1.1.
        2. If all images must be extracted (doesn't matter if it conatins a target class or not), get a list of all the available images' filenames.
            2.1. If only a limited number of images are to be extracted, sample a random batch of test images from the filenames extracted in step 2.
        3. Copy each of the selected image files in the new dataset's directory, and add each filename to the record.
        """

        # Initialize record lists and dataset_type variable (inside this function dataset_type can only be test)
        images_record = list()
        dataset_type = 'test'

        # If only test images with target classes can be extracted
        if self.test_only_target_classes:

            if os.path.exists(self.coco_annotation_test):
                with open(self.coco_annotation_test, 'r') as f:
                    self.coco_data = json.load(f)
                _, test_images_to_extract = self.extract_images_id_and_filenames(extract_target_images=True)
                if self.test_num_images:
                    test_images_to_extract = random.sample(test_images_to_extract, self.test_num_images)

            else:
                print(f'Only test images with target classes must be extracted, but the annotation file for the test set was not found in {self.coco_annotation_test}')

        # If all test images can be extracted
        else:
            test_images_to_extract = os.listdir(self.coco_image_dir_test) if not self.test_num_images else random.sample(os.listdir(self.coco_image_dir_test), self.test_num_images)

        # Copy the selected test images to the new dataset's directory, and add the filename to the record
        for filename in tqdm(test_images_to_extract):
            original_filename = os.path.join(self.coco_image_dir_test, filename)
            new_filename = os.path.join(self.output_dir, 'test', 'images', filename)
            shutil.copy(original_filename, new_filename)
            images_record.append(new_filename)
            
        # Save YOLOv8 or COCO format record of images filenames
        with open(os.path.join(self.output_dir, f'{dataset_type}.txt'), 'w') as dataset_list:
            dataset_list.write('\n'.join(images_record))

        print(f'{dataset_type.capitalize()} images successfully stored in {os.path.join(self.output_dir, f"{dataset_type}.txt")}. Total images: {len(images_record)}')

    def process_train_val_data(self,
                               dataset_type: str,
                               coco_annotation_file: str,
                               coco_image_dir: str,
                               single_class_name: Optional[str]) -> None:
        """
        Processes the train or validation data.
        1. Loads the original COCO annotation file.
        2. If a new class must be created, creates a new class.
        3. Extracts the IDs of the images to be extracted (all images or only those containing target classes).
            3.1. If background images must be added, extracts also the IDs of those background images.
        4. Convert and save the annotations from COCO to YOLO for every image extracted in step 3. Also, copy those images in the new directory.
        5. Save the record of the images and annotations included in the new dataset.
        Args:
            dataset_type (str): train/valid/test.
            coco_annotation_file (str): Path to the annotations of the original COCO dataset.
            coco_image_dir (str): Path to the directory which contains the images of the original COCO dataset.
            single_class_name (Optional[str]): Name of the single class to be generated, if specified by the create_single_class parameter.
        """

        with open(coco_annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        print(f'Number of images present in {dataset_type} data: {len([image_info["id"] for image_info in self.coco_data["images"]])}')

        # Create a new category for the single class if required
        if single_class_name:
            self.create_new_class()

        # Get unique image IDs with/out target classes if required
        unique_images_with_target_classes, unique_images_without_target_classes = set(), set()
        if self.target_classes:
            unique_images_with_target_classes, _ = self.extract_images_id_and_filenames(extract_target_images=True)
            if self.background_percentage > 0.0:
                unique_images_without_target_classes, _ = self.extract_images_id_and_filenames(extract_target_images=False,
                                                                              num_img_with_target_classes=len(unique_images_with_target_classes))

        else:
            unique_images_with_target_classes, _ = self.extract_images_id_and_filenames(extract_all=True)


        # Initialize record lists
        images_record, labels_record = list(), list()
        # Iterate through unique images with target classes
        if self.target_classes: print(f'Converting annotations that contain the target classes in {dataset_type} data...')
        else: print(f'Converting annotations for all images in {dataset_type} data...')
        self.convert_and_save_annotations_and_img(unique_images=unique_images_with_target_classes,
                                                  coco_image_dir=coco_image_dir,
                                                  dataset_type=dataset_type,
                                                  images_record=images_record,
                                                  labels_record=labels_record,
                                                  is_background=False)
            
        if unique_images_without_target_classes: 
            print(f'Converting annotations that do not have the target classes in {dataset_type} data... (Number of background images is {self.background_percentage}% of the images that contain target classes)')
            # Iterate through unique images without target classes
            self.convert_and_save_annotations_and_img(unique_images=unique_images_without_target_classes,
                                                      coco_image_dir=coco_image_dir,
                                                      dataset_type=dataset_type,
                                                      images_record=images_record,
                                                      labels_record=labels_record,
                                                      is_background=True)
        print(f'Annotations successfully converted for {dataset_type} data')
            
        # Save YOLOv8 or COCO format lists
        with open(os.path.join(self.output_dir, f'{dataset_type}.txt'), 'w') as dataset_list:
            dataset_list.write('\n'.join(images_record))

        print(f'{dataset_type.capitalize()} images and label data successfully stored in {os.path.join(self.output_dir, f"{dataset_type}.txt")}. Total images: {len(images_record)}')


    def convert_and_save_annotations_and_img(self,
                                             unique_images: set,
                                             coco_image_dir: str,
                                             dataset_type: str,
                                             images_record: List[str],
                                             labels_record: List[str],
                                             is_background: bool) -> None:
        """
        Iterates through the previously selected images (whose IDs are stored in unique_images parameter) and converts 
        the image's annotations to the YOLO format. It also saves the new annotations and copies the selected images in 
        the new dataest's directory.
        Args:
            unique_images (set): Set of image IDs, which correspond to the IDs of the images selected for extraction.
            coco_image_dir (str): path to the directory which contains the images of the original COCO dataset.
            dataset_type (str): train/valid/test
            images_record (List[str]): List which records all the new dataset's images' filenames.
            labels_record (List[str]): List which records all the new dataset's labels' filenames.
            is_background (bool): Boolean indicating whether the set of images IDs correspond to images containing 
                the target objects or just background.
        """

        # Iterate through unique images with target classes
        for img_id in tqdm(unique_images):
            img_info: Dict[str, Union[str, int]] = next((img for img in self.coco_data.get('images', []) if img['id'] == img_id), None)
                
            if img_info:

                # Copy image to the new directory
                img_path = os.path.join(coco_image_dir, img_info['file_name'])
                shutil.copy(img_path, os.path.join(self.output_dir, dataset_type, 'images'))
                images_record.append(os.path.join(dataset_type, 'images', img_info['file_name'])) # Record image filename

                if is_background:
                    # If it is a background image, no target class is present
                    label_content = ''

                else:
                    # Convert annotations to YOLOv8 or COCO format
                    annotations = [ann for ann in self.coco_data.get('annotations', []) if ann['image_id'] == img_id]
                    if self.convert_to_yolo:
                        label_content = self.convert_annotations_from_coco_to_yolo(img_info=img_info, annotations=annotations)
                    else:
                        # TODO: Add logic for COCO-COCO conversions
                        pass

                # Save YOLOv8 or COCO format label file
                label_filename = os.path.splitext(img_info['file_name'])[0] + ('.txt' if self.convert_to_yolo else '.json')
                label_filepath = os.path.join(self.output_dir, dataset_type, 'labels' if self.convert_to_yolo else 'annotations', label_filename)
                os.makedirs(os.path.dirname(label_filepath), exist_ok=True) # Ensure the directory exists before writing the label file
                with open(label_filepath, 'w') as label_file:
                    label_file.write(label_content)
                    
                # Record label filename
                labels_record.append(label_filepath)
    
    def convert_annotations_from_coco_to_yolo(self, 
                                              img_info: Dict[str, Union[int, str]],
                                              annotations: List[Dict[str, Union[List[List[float]], float, int, List[float]]]]) -> str:
        """
        Converts the received image's original annotations in COCO format to the YOLOv8 format.
        Args:
            img_info (Dict[str, Union[int, str]]): Image info extracted from the original dataset's data.
                The structure of the img_info variable is the following:
                    {
                        "license": int,
                        "file_name": str,
                        "coco_url": str,
                        "height": int,
                        "width": int,
                        "date_captured": str,
                        "flickr_url": str,
                        "id": int
                    }
            annotations (List[Dict[str, Union[List[List[float]], float, int, List[float]]]]): List of annotations for the image.
                The structure of every elements of the annotations array is the following:
                    {
                        "segmentation": [[float, ...]], (one nested array per segmentation annotation)
                        "area": float,
                        "iscrowd": int,
                        "image_id": int,
                        "bbox": [float, float, float, float],
                        "category_id": int,
                        "id": int
                    }
        Returns:
            str: Label (annotation) content in YOLOv8 format.
        """

        label_content = ""

        # Iterate through each annotation
        for ann in annotations:
            category_id = ann['category_id']
            category_name = next((cat['name'] for cat in self.coco_data.get('categories', []) if cat['id'] == category_id), None)
            
            # If the category of the annotation is a target one, include the annotation in YOLOv8 format
            if category_name in self.target_classes:
                category_id = self.target_classes.index(category_name) if not self.create_single_class else 0
                # COCO format: (x, y, width, height)
                bbox = ann['bbox']
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                width = bbox[2]
                height = bbox[3]

                # Normalize values to be between 0 and 1
                x_center /= img_info['width']
                y_center /= img_info['height']
                width /= img_info['width']
                height /= img_info['height']

                label_content += f"{category_id} {x_center} {y_center} {width} {height}\n"

        return label_content
    
    def extract_images_id_and_filenames(self,
                                        extract_target_images: bool = True,
                                        num_img_with_target_classes: Optional[int] = None,
                                        extract_all: bool = False) -> Tuple[Set[int]]:
        """
        Extracts the image IDs and filenames of those images who must be extracted to the new dataset.
        Args:
            extract_target_images (bool): If set to True, only the IDs and filenames of images that contain the target images will be extracted.
                                          If set to False, only the IDs and filenames of images that do not contain the target images will be extracted.
            num_img_with_target_classes (Optional[int]): This parameter represents the number of unique images containing the target 
                classes that currently are in the new dataset. It is used to calculate the number of background images that must be 
                extracted as a percentage of the current number of target class images.
            extract_all (bool): If set to True, all image IDs and filenames are extracted.
        Returns:
            Tuple[Set[int]]: Tuple containing the sets of the IDs anf filenames of the images that must be moved and converted to 
                the new dataset, according to the user's configuration. 
        """

        unique_images_id, unique_images_filenames = set(), set()

        if extract_all:
            return set([image_info['id'] for image_info in self.coco_data['images']]), \
                   set([image_info['file_name'] for image_info in self.coco_data['images']])

        # Iterate through annotations to search for target/non-target images
        for ann in self.coco_data.get('annotations', []):
            category_id = ann['category_id']
            category_name = next((cat['name'] for cat in self.coco_data.get('categories', []) if cat['id'] == category_id), None)
            
            # Check if the annotation is for any target class
            if (category_name in self.target_classes) == extract_target_images:
                image_id = ann['image_id']

                # Add ID to set
                unique_images_id.add(image_id)
                # Add filename toset
                result_dict = next((img_data for img_data in self.coco_data['images'] if img_data['id'] == image_id), None)
                unique_images_filenames.add(result_dict['file_name'])

                if not extract_target_images:
                    if len(unique_images_id) > num_img_with_target_classes * self.background_percentage:
                        break

        return unique_images_id, unique_images_filenames
    
    def create_new_class(self) -> None:
        """Creates a new class in the original annotations file"""
        # Create a new category for the single class
        new_category_id = next((cat['id'] for cat in self.coco_data.get('categories', []) if cat['name'] == self.single_class_name), None)
        if new_category_id is None:
            new_category_id = max(cat['id'] for cat in self.coco_data.get('categories', [])) + 1
            new_category = {
                'id': new_category_id,
                'name': self.single_class_name,
                'supercategory': self.single_class_name,
            }
            self.coco_data.setdefault('categories', []).append(new_category)

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parses the arguments received from the user and sets this values as class attributes.
        """

        # Create an ArgumentParser object
        # TODO: fill text
        parser = argparse.ArgumentParser(description='Description of your script.')

        # Add arguments
        parser.add_argument('dataset_dir', type=str, help='Path to the directory where COCO dataset is located.')
        parser.add_argument('--output_dir', type=str, default='new_dataset', help='Name of the directory where the new dataset will be generated.')
        parser.add_argument('--target_classes', '--names-list', nargs='+', default=[], help='Array of strings,where each string is the name of the '
                                                                                            'class whose images that must be extracted from the original COCO dataset.')
        parser.add_argument('--background_percentage', type=float, default=0.0, help='Only applies if some classes are being extracted from COCO dataset. '
                            'The new dataset will include <background_percentage>% more images, which will not contain any of the target classes.')
        parser.add_argument('--test_num_images', type=int, help='Number of test images from the original COCO dataset to include in the new dataset.')
        parser.add_argument('--test_only_target_classes', type=bool, default=False, help='Boolean indicating whether to include only images with the target classes or any image.')
        parser.add_argument('--create_single_class', type=bool, default=False, help='Boolean indicating whether to join all the selected classes into a single class. Defaults to True.')
        parser.add_argument('--single_class_name', type=str, default='new_class', help='Only applies if create_single_class param is set to True. Name of the single class to be generated.')
        parser.add_argument('--convert_to_yolo', type=bool, default=True, help='Boolean indicating whether to convert the annotations to YOLOv8 or not.')

        # Parse the command line arguments
        args = parser.parse_args()

        print('PARAMETERS =================================================================================\n')
        print(f"Dataset Directory: {args.dataset_dir}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Target Classes: {args.target_classes}")
        print(f"Background Percentage: {args.background_percentage}")
        print(f"Test Number of Images: {args.test_num_images}")
        print(f"Test Only Target Classes: {args.test_only_target_classes}")
        print(f"Create Single Class: {args.create_single_class}")
        print(f"Single Class Name: {args.single_class_name}")
        print(f"Convert to YOLO: {args.convert_to_yolo}")

        self.coco_annotation_train = os.path.join(args.dataset_dir, 'annotations', 'instances_train2017.json')
        self.coco_image_dir_train = os.path.join(args.dataset_dir, 'images', 'train2017')
        self.coco_annotation_val = os.path.join(args.dataset_dir, 'annotations', 'instances_val2017.json')
        self.coco_image_dir_val = os.path.join(args.dataset_dir, 'images', 'val2017')
        self.coco_annotation_test = os.path.join(args.dataset_dir, 'annotations', 'instances_test2017.json')
        self.coco_image_dir_test = os.path.join(args.dataset_dir, 'images', 'test2017')
        self.output_dir = args.output_dir
        self.target_classes = args.target_classes
        self.background_percentage = args.background_percentage
        self.test_num_images = args.test_num_images
        self.test_only_target_classes = args.test_only_target_classes
        self.create_single_class = args.create_single_class
        self.single_class_name = args.single_class_name
        self.convert_to_yolo = args.convert_to_yolo

    @staticmethod
    def serialize(obj):
        if isinstance(obj, (set,)):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, range):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    coco_converter = COCOConverter()
    coco_converter.initialize_conversion()

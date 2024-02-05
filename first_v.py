def coco_to_yolo_format(img_info, 
                        target_classes,
                        annotations, 
                        coco_data, 
                        single_class_name=None):
    
    label_content = ""

    # Iterate through each annotation
    for ann in annotations:
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in coco_data.get('categories', []) if cat['id'] == category_id), None)
        
        # If the category of the annotation is a target one, include the annotation in YOLOv8 format
        if category_name in target_classes:
            category_id = target_classes.index(category_name) if single_class_name else 0
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

def extract_images_id(coco_data: Any,
                      target_classes: List[str],
                      extract_target_images: bool = True,
                      num_img_with_target_classes: Optional[int] = None,
                      background_img_percentage: Optional[float] = None) -> set:
    unique_images = set()

    # Iterate through annotations to search for target/non-target images
    for ann in tqdm(coco_data.get('annotations', [])):
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in coco_data.get('categories', []) if cat['id'] == category_id), None)

        # Check if the annotation is for any target class
        if (category_name in target_classes) == extract_target_images:
            image_id = ann['image_id']
            unique_images.add(image_id)
            if not extract_target_images:
                if len(unique_images) > num_img_with_target_classes * background_img_percentage:
                    break

    return unique_images

def create_new_class(single_class_name: str,
                     coco_data: Any) -> None:
    # Create a new category for the single class
    new_category_id = next((cat['id'] for cat in coco_data.get('categories', []) if cat['name'] == single_class_name), None)
    if new_category_id is None:
        new_category_id = max(cat['id'] for cat in coco_data.get('categories', [])) + 1
        new_category = {
            'id': new_category_id,
            'name': single_class_name,
            'supercategory': single_class_name,
        }
        coco_data.setdefault('categories', []).append(new_category)

def convert_annotations(unique_images: set,
                        coco_data: Any,
                        coco_image_dir: str,
                        output_dir: str,
                        dataset_type: str,
                        convert_to_yolo_format: bool,
                        target_classes: List[str],
                        images_record: List[str],
                        labels_record: List[str],
                        is_background: bool) -> List[str]:
    # Iterate through unique images with target classes
    for img_id in tqdm(unique_images):
        img_info = next((img for img in coco_data.get('images', []) if img['id'] == img_id), None)
            
        if img_info:    
            # Copy image to the new directory
            img_path = os.path.join(coco_image_dir, img_info['file_name'])
            shutil.copy(img_path, os.path.join(output_dir, dataset_type, 'images'))
            images_record.append(os.path.join(dataset_type, 'images', img_info['file_name'])) # Record image filename

            if not is_background:
                # Convert annotations to YOLOv8 or COCO format
                annotations = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] == img_id]
                if convert_to_yolo_format:
                    label_content = coco_to_yolo_format(img_info=img_info, target_classes=target_classes, 
                                                        annotations=annotations, coco_data=coco_data, single_class_name=single_class_name)
                else:
                    label_content = json.dumps({'annotations': annotations})

            # Save YOLOv8 or COCO format label file
            label_filename = os.path.splitext(img_info['file_name'])[0] + ('.txt' if convert_to_yolo_format else '.json')
            label_filepath = os.path.join(output_dir, dataset_type, 'labels' if convert_to_yolo_format else 'annotations', label_filename)
            os.makedirs(os.path.dirname(label_filepath), exist_ok=True) # Ensure the directory exists before writing the label file
            with open(label_filepath, 'w') as label_file:
                label_file.write(label_content)
                
            # Record label filename
            labels_record.append(label_filepath)

def process_dataset(dataset_type: str, 
                    coco_annotation_file: str, 
                    background_img_percentage: float,
                    coco_image_dir: str, 
                    output_dir: str, 
                    target_classes: List[str],
                    single_class_name: Optional[str] = None,
                    convert_to_yolo_format: bool = True) -> None:
    """
    Processes the new dataset, which will be a subset of the COCO dataset with 
    only the images corresponding to the classes given by the target_classes parameter, 
    and their new labels. Note that the new labels have as many classes as the length of 
    target_classes array.
    Finally, this new dataset is converted to YOLOv8 format.
    Args:
        dataset_type (str): train/valid/test.
        coco_annotation_file (str): path to the annotations of the original COCO dataset.
        coco_image_dir (str): path to the directory which contains the images of the original COCO dataset.
        output_dir (str): directory where the new dataset will be generated.
        target_classes (List[str]): list of the classes that should be kept. All classes must exist in the 
        original COCO dataset.
    """
    # Load COCO annotations if they exist
    if os.path.exists(coco_annotation_file):
        with open(coco_annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Create a new category for the single class if required
        if single_class_name:
            create_new_class(single_class_name=single_class_name,
                             coco_data=coco_data)

        # Get unique image IDs with/out target classes
        unique_images_with_target_classes = extract_images_id(coco_data=coco_data,
                                                              target_classes=target_classes,
                                                              extract_target_images=True)
        unique_images_without_target_classes = extract_images_id(coco_data=coco_data,
                                                                 target_classes=target_classes,
                                                                 extract_target_images=False,
                                                                 num_img_with_target_classes=len(unique_images_with_target_classes),
                                                                 background_img_percentage=background_img_percentage)

        # Initialize record lists
        images_record, labels_record = list(), list()
        # Iterate through unique images with target classes
        print('Iterate through unique images with target classes')
        convert_annotations(unique_images=unique_images_with_target_classes,
                            coco_data=coco_data,
                            coco_image_dir=coco_image_dir,
                            output_dir=output_dir,
                            dataset_type=dataset_type,
                            convert_to_yolo_format=convert_to_yolo_format,
                            target_classes=target_classes,
                            images_record=images_record,
                            labels_record=labels_record,
                            is_background=False)
        num_img_with_target_class = len(images_record)
        print(f'Iteration through unique images with target classes finished. Number of images with target classes: {num_img_with_target_class}')
        
        # Iterate through unique images without target classes
        print('Iterate through unique images without target classes')
        convert_annotations(unique_images=unique_images_without_target_classes,
                            coco_data=coco_data,
                            coco_image_dir=coco_image_dir,
                            output_dir=output_dir,
                            dataset_type=dataset_type,
                            convert_to_yolo_format=convert_to_yolo_format,
                            target_classes=target_classes,
                            images_record=images_record,
                            labels_record=labels_record,
                            is_background=True)
        print(f'Iteration through unique images with target classes finished. Number of images without target classes: {len(images_record) - num_img_with_target_class}')
        
        # Save YOLOv8 or COCO format lists
        with open(os.path.join(output_dir, f'{dataset_type}.txt'), 'w') as dataset_list:
            dataset_list.write('\n'.join(images_record))

    else:
        print(f'Could not find a COCO-format annotation file in {coco_annotation_file}')

def main() -> None:    
    
    args = parse_arguments()
    
    # TODO: create parameter
    convert_to_yolo_format = True

    # Set the paths for the COCO dataset annotations and images
    coco_annotation_train = os.path.join(args.arg1, 'annotations', 'instances_train2017.json')
    coco_image_dir_train = os.path.join(args.arg1, 'images', 'train2017')
    coco_annotation_val = os.path.join(args.arg1, 'annotations', 'instances_val2017.json')
    coco_image_dir_val = os.path.join(args.arg1, 'images', 'val2017')

    # Define the classes that must be extracted
    target_classes = args.arg3

    # Set the output directory for the new dataset
    output_dir = args.arg2

    # Create output directories
    for dataset_type in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, dataset_type, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, dataset_type, 'labels'), exist_ok=True)

    # Process each dataset
    process_dataset('train', 
                    coco_annotation_train, 
                    args.arg4, 
                    coco_image_dir_train, 
                    output_dir, 
                    target_classes, 
                    args.arg8 if args.arg7 else None,
                    convert_to_yolo_format)
    process_dataset('valid', 
                    coco_annotation_val, 
                    args.arg4, 
                    coco_image_dir_val, 
                    output_dir, 
                    target_classes, 
                    args.arg8 if args.arg7 else None,
                    convert_to_yolo_format)


if __name__ == '__main__':
    main()
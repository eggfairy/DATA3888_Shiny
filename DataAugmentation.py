import cv2
import random
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt

# def data_augmentation(img):
#     augmentation_list = []
#     for i in range(4):
#         augmentation_list.append(img)
#         img = cv2.rotate(img, 90, cv2.ROTATE_90_CLOCKWISE)

#     img = cv2.flip(img, 1)

#     for i in range(4):
#         augmentation_list.append(img)
#         img = cv2.rotate(img, 90, cv2.ROTATE_90_CLOCKWISE)
    
#     return augmentation_list

# def simple_augmentation(img):
#     ops = [
#         lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
#         lambda x: cv2.rotate(x, cv2.ROTATE_180),
#         lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
#         lambda x: cv2.flip(x, 0),  # Flip Vertically
#         lambda x: cv2.flip(x, 1),  # Flip Horizontal
#     ]
#     op = random.choice(ops)
#     return op(img)
# def augment_dataset(images, labels):
#     from collections import Counter
#     label_counter = Counter(labels)
#     max_count = max(label_counter.values())
#     augmented_images = []
#     augmented_labels = []

#     for img, label in zip(images, labels):
#         current_count = label_counter[label]
#         num_to_add = max_count - current_count

#         augmented_images.append(img)
#         augmented_labels.append(label)
        
#         for _ in range(num_to_add):
#             aug_img = simple_augmentation(img)
#             augmented_images.append(aug_img)
#             augmented_labels.append(label)
            
#     return augmented_images, augmented_labels


# Data-pre arguring:
#dataset = 100
def group_images_by_brightness(dataset, n_clusters=5, show_summary=True):
    def get_brightness(img):
        if len(img.shape) == 3:
            return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            return np.mean(img)

    brightness_values = np.array([get_brightness(img) for img in dataset]).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(brightness_values)
    grouped_data = defaultdict(list)
    
    for img, label in zip(dataset, labels):
        grouped_data[label].append(img)

    if show_summary:
        plt.figure(figsize=(8, 4))
        plt.hist(brightness_values, bins=30, color='gray')
        plt.title("Brightness Distribution")
        plt.xlabel("Average Brightness")
        plt.ylabel("Number of Images")
        plt.grid(True)
        plt.show()

    return grouped_data, labels
#group_0_images = grouped_images[0]


#If need to do comparison between models
#def set_augmentation_seed(seed):
#    random.seed(seed)
#    np.random.seed(seed)
#    A.set_seed(seed)

def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomCrop(height=224, width=224, p=1.0),
    ])
    
def augment_image(img):
    augmenter = get_augmentation_pipeline()
    augmented = augmenter(image=img)
    return augmented['image']

def balance_dataset_with_augmentation(images, labels, target_size=None):
    label_counter = Counter(labels)
    
    if target_size is None:
        target_size = max(label_counter.values())
    
    new_images = []
    new_labels = []

    for label in label_counter:
        label_imgs = [img for img, lbl in zip(images, labels) if lbl == label]
        current_count = label_counter[label]
        
        new_images.extend(label_imgs)
        new_labels.extend([label] * current_count)
        
        num_to_add = target_size - current_count
        
        for _ in range(num_to_add):
            img = random.choice(label_imgs)
            aug_img = augment_image(img)
            new_images.append(aug_img)
            new_labels.append(label)
            
    return new_images, new_labels

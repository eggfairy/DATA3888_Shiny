import cv2 

def data_augmentation(img):
    augmentation_list = []
    for i in range(4):
        augmentation_list.append(img)
        img = cv2.rotate(img, 90, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.flip(img, 1) #flip on y axis

    for i in range(4):
        augmentation_list.append(img)
        img = cv2.rotate(img, 90, cv2.ROTATE_90_CLOCKWISE)
    
    return augmentation_list

import os
import shutil
import glob

source_path = './Images/100/'
folder1 = input("The first folder you want to merge: ")
source_folder_path1 = source_path + folder1
if not os.path.isdir(source_folder_path1):
    print(f'the folder does not exist{source_folder_path1}')
    os._exit()

folder2 = input("The second folder you want to merge: ")
source_folder_path2 = source_path + folder2
if not os.path.isdir(source_folder_path2):
    print(f'the folder does not exist{source_folder_path2}')
    os._exit()
folder_dest = input("The destination folder you want to merge: ")
dest_path = source_path + folder_dest
os.makedirs(dest_path, exist_ok=True)

source_dirs = [
    source_folder_path1,
    source_folder_path2,
]

extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
count = 1
for src in source_dirs:
    for ext in extensions:
        pattern = os.path.join(src, ext)
        for img_path in glob.glob(pattern):
            filename = os.path.basename(img_path)
            file_dest_path = os.path.join(dest_path, filename)
            shutil.copy2(img_path, file_dest_path)
            print(f'{count}: Copied {img_path} -> {file_dest_path}')
            count += 1
print('Complete!')

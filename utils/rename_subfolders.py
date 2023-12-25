import os


# Rename subfolders within the dataset to replace spaces with underscores
# so that later we can work with paths easier, e.g. paths with white spaces gave me problems
# when reading them from a dataframe and trying to read from them

directory_path = '/home/juancm/trento/SIV/siv_project/dataset_backup_13-12-2023/Powerlifting_Dataset'

for root, dirs, files in os.walk(directory_path):
    for d in dirs:
        new_name = d.replace(' ', '_')
        new_name = new_name.replace(' - ', '_')
        os.rename(os.path.join(root, d), os.path.join(root, new_name))
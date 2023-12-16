import os
import pandas as pd

# Path to your dataset
dataset_path = "/home/juancm/trento/SIV/siv_project/dataset_backup_13-12-2023/Powerlifting_Dataset"

image_labels = []

# Traverse the directory structure
for root, dirs, files in os.walk(dataset_path):
    # The subfolder name is the label
    label = os.path.basename(root)
    
    # for each file in the subfolder
    for file in files:
        # Only consider image files
        if file.endswith('.jpg') or file.endswith('.png'):
            if not file.startswith('Screenshot'):
                # Get the paths from root
                path_from_root = os.path.relpath(os.path.join(root, file), dataset_path).replace(" ","\ ")
                # Add the file and label to the list
                image_labels.append((file, label, path_from_root))

# Convert the list to a DataFrame
df = pd.DataFrame(image_labels, columns=["image", "label", "relative path"])

# remap old labels into new ones: above, parallel, valid
new_labels = {'Frontal_Above_parallel':'above', 'Lateral_Above_parallel':'above',
              'Frontal_Parallel':'parallel', 'Lateral_Parallel':'parallel',
              'Frontal_Valid':'valid', 'Lateral_Valid':'valid'}

df['label'] = df['label'].replace(new_labels)
# Save the DataFrame as a CSV file
df.to_csv("image_labels.csv", index=False)
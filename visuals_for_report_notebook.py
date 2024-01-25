#%%
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#%%
# Define your custom dataset path and image filename
image_path = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/dataset/Men/Original/White men 120kg/WM120/Frontal - Valid/6WM120_3.jpg'
# Load the image using PIL
image = Image.open(image_path)

#%% Apply the different transformations to the same image
if image.mode != 'RGB':
    image = image.convert('RGB')

# Define the transformations
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomPerspective(distortion_scale=0.2, p=1.0, fill=0),
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomGrayscale(p=1.),
])

transform3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Apply the transformations
image_transformed1 = transform1(image)
image_transformed2 = transform2(image)
image_transformed3 = transform3(image)
#%%
# Plot the transformed images in a 1x4 grid
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('RandomPerspective')
plt.imshow(image_transformed1)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('RandomGrayscale')
plt.imshow(image_transformed2)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('ColorJitter')
plt.imshow(image_transformed3)
plt.axis('off')

plt.show()
# %% load frontal images and preprocess them for plotting
image_path_above = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/above.png'
image_path_parallel = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/parallel.png'
image_path_valid = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/valid.png'

image_above = Image.open(image_path_above)
image_parallel = Image.open(image_path_parallel)
image_valid = Image.open(image_path_valid)
if image_above.mode != 'RGB':
    image_above = image_above.convert('RGB')
if image_parallel.mode != 'RGB':
    image_parallel = image_parallel.convert('RGB')
if image_valid.mode != 'RGB':
    image_valid = image_valid.convert('RGB')

above = transforms.Resize((224, 224))(image_above)
parallel = transforms.Resize((224, 224))(image_parallel)
valid = transforms.Resize((224, 224))(image_valid)

#%% load lateral images and preprocess them for plotting
image_path_above_lateral = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/above_lateral.png'
image_path_parallel_lateral = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/parallel_lateral.png'
image_path_valid_lateral = '/home/juancm/trento/SIV/siv_project/Back_Squats_IPF/report_images/valid_lateral.png'

image_above_lateral = Image.open(image_path_above_lateral)
image_parallel_lateral = Image.open(image_path_parallel_lateral)
image_valid_lateral = Image.open(image_path_valid_lateral)
if image_above_lateral.mode != 'RGB':
    image_above_lateral = image_above_lateral.convert('RGB')
if image_parallel_lateral.mode != 'RGB':
    image_parallel_lateral = image_parallel_lateral.convert('RGB')
if image_valid_lateral.mode != 'RGB':
    image_valid_lateral = image_valid_lateral.convert('RGB')

above_lateral = transforms.Resize((224, 224))(image_above_lateral)
parallel_lateral = transforms.Resize((224, 224))(image_parallel_lateral)
valid_lateral = transforms.Resize((224, 224))(image_valid_lateral)



#%% Plot the images in a 2x3 grid
plt.figure(figsize=(12, 12))

plt.subplot(2, 4, 1)
plt.title('Above parallel - Frontal')
plt.imshow(above)
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title('Parallel - Frontal')
plt.imshow(parallel)
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title('Valid - Frontal')
plt.imshow(valid)
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('Above parallel - Lateral')
plt.imshow(above_lateral)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Parallel - Lateral')
plt.imshow(parallel_lateral)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Valid - Lateral')
plt.imshow(valid_lateral)
plt.axis('off')

plt.show()
# %%

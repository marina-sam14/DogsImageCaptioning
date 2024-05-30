import os
import shutil
import pandas as pd
from PIL import Image
from io import BytesIO

# Install Git Large File Storage (LFS)
def install_git_lfs():
    os.system("git lfs install")

# Clone a repository 
def clone_repository(repo_url, clone_dir):
    os.system(f"git clone {repo_url} {clone_dir}")

# Set the environment variables
def set_environment_variables():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # gpu

# Read the Parquet file
def read_parquet_file(parquet_file_path):
    return pd.read_parquet(parquet_file_path)

# Save the images to the output folder
def save_images_from_dataframe(df, output_folder):
    # Create a folder to save the images, deleting it first if it already exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for index, row in df.iterrows():
        # Extract image data
        image_data = row['image']['bytes']

        # Open image using PIL
        image = Image.open(BytesIO(image_data))

        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_filename = f'image_{index}.jpg'
        image.save(os.path.join(output_folder, image_filename))

    print("Images saved successfully in the 'images' folder")

def update_dataframe_with_paths(df, output_folder):
    df['filename'] = df.index.map(lambda x: f'image_{x}.jpg')
    df['path'] = df['filename'].apply(lambda x: os.path.join(output_folder, x))
    return df

def display_first_image(image_path):
    image = Image.open(image_path)
    image.show()

# Main function
def main():
    install_git_lfs() 
    clone_repository("https://huggingface.co/datasets/fusing/dog_captions", "dog_captions")  
    set_environment_variables()  
    
    parquet_file_path = '/home/msamprovalaki/captioning/dog_captions/data/train-00000-of-00002.parquet' 
    df = read_parquet_file(parquet_file_path) 
    
    output_folder = 'images'  
    save_images_from_dataframe(df, output_folder) 
    
    df = update_dataframe_with_paths(df, output_folder)  
    print(df) 
    
    first_image_path = df['path'].iloc[0] 
    display_first_image(first_image_path)  

if __name__ == "__main__":
    main()

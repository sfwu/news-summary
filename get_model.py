import os
import gdown

def download_model(folder_id, local_dir):

    # Check if the local directory exists
    if os.path.exists(local_dir):
        return
    else:
        os.makedirs(local_dir)
        try:
            print(f"Downloading folder with ID '{folder_id}' to '{local_dir}'...")
            gdown.download_folder(id=folder_id, output=local_dir, quiet=False)
            print("Download completed!")
        except Exception as e:
            print("An error occurred during the download:", e)

if __name__ == "__main__":
    # Replace 'YOUR_FOLDER_ID' with your actual Google Drive folder ID
    folder_id = '1KGfBg_uCMg5AnzNcAojAhbqiO13gw6u6'
    local_dir = 'fine_tuned_model'
    download_model(folder_id, local_dir)

import json
import glob
import pandas as pd
from PIL import Image

def get_annotations(id):
    annotation_path = f"../../../mnt/nfs_mount/single_frames/{id}/annotations/object_detection/"
    annotation_file = glob.glob(annotation_path + "*.json")
    f = open(annotation_file[0])
    annotation_list = json.load(f)

    column_names = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
    df = pd.DataFrame(columns=column_names)

    # load image to get shape
    image_path = f"../../../mnt/nfs_mount/single_frames/{id}/camera_front_blur/"
    image_files = glob.glob(image_path + "*.jpg")
    if len(image_files) > 0:
        image = Image.open(image_files[0])
    size = image.size

    # only get cars
    for annotation in annotation_list:
        if annotation["properties"]["class"] == "Vehicle":
            # get bounding boxes
            box_coordinates = annotation["geometry"]["coordinates"]

            # Rescale bounding boxes 
            new_size = (224, 224)
            scale_x = new_size[0] / size[0]
            scale_y = new_size[1] / size[1]
            
            min_x = min(coord[0] for coord in box_coordinates) * scale_x
            min_y = min(coord[1] for coord in box_coordinates) * scale_y
            max_x = max(coord[0] for coord in box_coordinates) * scale_x
            max_y = max(coord[1] for coord in box_coordinates) * scale_y

            new_row = {"image_id": id, "x_min": min_x, "y_min": min_y, "x_max": max_x, "y_max": max_y}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def create_df_bounding_boxes():
    # The parent directory containing the folders
    parent_directory = "../../../mnt/nfs_mount/single_frames"

    # Find all folders with a 6-digit name
    folder_pattern = os.path.join(parent_directory, "[0-9]" * 6)

    # Get the list of matching folders
    folders = glob.glob(folder_pattern)
    folders = folders[:5000] # first 5000 folders

    column_names = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
    total_df = pd.DataFrame(columns=column_names)

    # iterate through all folders (009999 etc.)
    for folder in folders:
        id = os.path.basename(folder) # id = foldername
        df = get_annotations(id)
        total_df = pd.concat([total_df, df], ignore_index=True)
    return total_df


if __name__=="__main__":
    df = create_df_bounding_boxes()
    df.to_csv('df_bounding_boxes.csv')
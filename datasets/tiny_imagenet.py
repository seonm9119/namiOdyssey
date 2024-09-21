import os
import pandas as pd
from glob import glob

_classes_=['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172',
 'n01768244', 'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01855672',
 'n01882714', 'n01910747', 'n01917289', 'n01944390', 'n01945685', 'n01950731',
 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221', 'n02074367',
 'n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799',
 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136',
 'n02165456', 'n02190166', 'n02206856', 'n02226429', 'n02231487', 'n02233338',
 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529', 'n02364673',
 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312',
 'n02480495', 'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666196',
 'n02669723', 'n02699494', 'n02730930', 'n02769748', 'n02788148', 'n02791270',
 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860',
 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205',
 'n02892201', 'n02906734', 'n02909870', 'n02917067', 'n02927161', 'n02948072',
 'n02950826', 'n02963159', 'n02977058', 'n02988304', 'n02999410', 'n03014705',
 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240', 'n03126707',
 'n03160309', 'n03179701', 'n03201208', 'n03250847', 'n03255030', 'n03355925',
 'n03388043', 'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034',
 'n03447447', 'n03544143', 'n03584254', 'n03599486', 'n03617480', 'n03637318',
 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131', 'n03763968',
 'n03770439', 'n03796401', 'n03804744', 'n03814639', 'n03837869', 'n03838899',
 'n03854065', 'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156',
 'n03976657', 'n03977966', 'n03980874', 'n03983396', 'n03992509', 'n04008634',
 'n04023962', 'n04067472', 'n04070727', 'n04074963', 'n04099969', 'n04118538',
 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777',
 'n04259630', 'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186',
 'n04356056', 'n04366367', 'n04371430', 'n04376876', 'n04398044', 'n04399382',
 'n04417672', 'n04456115', 'n04465501', 'n04486054', 'n04487081', 'n04501370',
 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935',
 'n04596742', 'n04597913', 'n06596364', 'n07579787', 'n07583066', 'n07614500',
 'n07615774', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07734744',
 'n07747607', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807',
 'n07875152', 'n07920052', 'n09193705', 'n09246464', 'n09256479', 'n09332890',
 'n09428293', 'n12267677']


_classes_.sort()
classes_to_idx = {label: idx for idx, label in enumerate(_classes_)}

def make_train_dataframe(root_dir, csv_path='train.csv'):
    """
    Merges all training data (txt files) from the given root_dir into a single DataFrame.

    Args:
        root_dir (str): The top-level directory path of the data.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    # Find all .txt files in the class directories
    txt_files = glob(os.path.join(root_dir, 'train', '*', '*.txt'))

    # Read txt files and merge into a DataFrame
    dataframes = []  # List to store DataFrames for each file
    for file_path in txt_files:
        class_folder = os.path.basename(os.path.dirname(file_path))
        image_dir = os.path.join(root_dir, 'train', class_folder, 'images')
        
        # Read the file and create a DataFrame with path information
        df = pd.read_csv(file_path, sep='\t', header=None, names=['image_path', 'x1', 'y1', 'x2', 'y2'])
        df['image_path'] = df['image_path'].apply(lambda img_name: os.path.join(image_dir, img_name))
        
        # Add the 'class_folder' column
        df['classes'] = class_folder
        df['label'] = df['classes'].map(classes_to_idx)
        
        # Add the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    train_df = pd.concat(dataframes, ignore_index=True)
    print(f"Saving merged training data to CSV file at: {csv_path}")
    train_df.to_csv(csv_path, index=False)

    return train_df


def make_val_dataframe(root_dir, csv_path='val.csv'):
    """
    Merges all validation data (txt files) from the given root_dir into a single DataFrame and saves it as a CSV file.

    Args:
        root_dir (str): The top-level directory path of the data.
        csv_path (str): Path where the merged CSV file will be saved. Default is 'val.csv'.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    # Find all .txt files in the validation directories
    txt_files = glob(os.path.join(root_dir, 'val', '*.txt'))

    # Read and process each txt file
    for file_path in txt_files:
        # Read the file and create a DataFrame with path and label information
        val_df = pd.read_csv(file_path, sep='\t', header=None, names=['image_path', 'classes', 'x1', 'y1', 'x2', 'y2'])
        
        # Update the 'image_path' column with the full path
        val_df['image_path'] = val_df['image_path'].apply(lambda img_name: os.path.join(root_dir, 'val', 'images',img_name))
    

    val_df['label'] = val_df['classes'].map(classes_to_idx)
    # Reorder columns to 'image_path', 'x1', 'y1', 'x2', 'y2', 'label'
    val_df = val_df[['image_path', 'x1', 'y1', 'x2', 'y2', 'classes', 'label']]

    # Save the merged validation data to a CSV file
    print(f"Saving merged validation data to CSV file at: {csv_path}")
    val_df.to_csv(csv_path, index=False)
    
    return val_df


SPLIT_FUNC = {'train':('train.csv', make_train_dataframe),
              'val': ('val.csv', make_val_dataframe)}

def check_file(data_dir, split='train'):
    """Helper function to check file existence and create if needed."""

    file, function = SPLIT_FUNC[split]
    file_path = os.path.join(data_dir, file)
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Creating the CSV file...")
        function(data_dir, csv_path=file_path)
    
    return file_path

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg


class TinyImageNet(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        super(TinyImageNet, self).__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        file_path = check_file(self.data_dir, split)
        df = pd.read_csv(file_path, usecols=['image_path', 'label'])
        self.img, self.label = df['image_path'], df['label']

    
    def __len__(self):
        return len(self.img)


    def __getitem__(self, idx):
        img = self.loader(self.img[idx])

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return {'input': img, 'label': label}


def build_loader(config, transform):

    loader = {}
    for split in SPLIT_FUNC.keys():
        datasets = TinyImageNet(config.data_dir, split, transform=transform[split])

        loader[split] = torch.utils.data.DataLoader(datasets, 
                                                    batch_size=config.batch_size,
                                                    shuffle=True if split == 'train' else False)

    return loader

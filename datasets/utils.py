import os

def check_file(data_dir, file, create_function):
    """Helper function to check file existence and create if needed."""

    file_path = os.path.join(data_dir, file)
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Creating the CSV file...")
        create_function(data_dir, csv_path=file_path)
    
    return file_path
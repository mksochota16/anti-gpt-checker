import os
from dao.lab_report import DAOLabReport
from models.lab_report import LabReport

dao = DAOLabReport()
def save_text_files_to_mongo(directory_path: str, is_generated: bool, model_name: str):
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a .txt extension
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            # Open and read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                non_blank_lines = [line.strip() for line in content if line.strip()]
                content = '\n'.join(non_blank_lines)
                # Save the content to MongoDB
                lab_report = LabReport(
                    plaintext_content=content,
                    tag=filename,
                    is_generated=is_generated,
                    model=model_name
                )
                dao.insert_one(lab_report)


if __name__ == "__main__":
    is_generated_input = input("Are the lab reports generated? (y/n): ")
    is_generated_input = is_generated_input.lower() == 'y'

    if is_generated_input:
        model_name = input("Which model was used? ")
    else:
        model_name = None

    path = input("Enter the path to the directory containing the lab reports: ")
    save_text_files_to_mongo(path, is_generated_input, model_name)
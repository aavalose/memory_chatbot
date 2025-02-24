import os


def qa_dict_gen(folder_path):
    """Generate a dictionary of questions and answers from text files in a folder.

    Args:
        folder_path (str): Path to folder containing Q&A text files

    Returns:
        dict: Dictionary mapping questions to their answers
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty dictionary to store all Q&As
    qa_dict = {}

    # Iterate over each file and extract Q&As
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Split content by Q: and A:
            parts = content.split("Q:")
            if len(parts) > 1:  # If Q: exists
                for part in parts[1:]:  # Skip first empty part
                    q_text = part.split("A:")[0].strip()
                    if "A:" in part:
                        a_text = part.split("A:")[1].strip()
                        qa_dict[q_text] = a_text

    return qa_dict
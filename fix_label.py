"""
Convert yolo image
"""

import os
LABEL_DIR = "nrc_images_yolo"
NEW_LABEL_DIR = 'fix_nrc_labels'

os.makedirs(NEW_LABEL_DIR, exist_ok=True)
for txt_file in os.listdir(LABEL_DIR):
    if txt_file.endswith('txt'):
        new_txt_file = os.path.join(NEW_LABEL_DIR, txt_file)
        with open(os.path.join(LABEL_DIR, txt_file), 'r')as f:
            with open(new_txt_file, "a") as new_f:
                lines = f.readlines()
                for line in lines:
                    if len(line) > 0:
                        line = list(line)
                        line[0] = str(int(line[0]) - 1)
                        line = "".join(line)
                        new_f.write(line)

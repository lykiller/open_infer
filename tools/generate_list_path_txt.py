import os
import sys

Path = r"G:\info\20240227"
Extensions = "info"
Title = "path"

path_list = os.listdir(Path)
Path = Path + "\\"
text_name = Path + Title
with open("%s.txt" % text_name, "w", encoding='GB2312') as f:
    for filename in path_list:
        if os.path.splitext(filename)[1] == r"." + Extensions:
            f.write(Path + filename + '\n')

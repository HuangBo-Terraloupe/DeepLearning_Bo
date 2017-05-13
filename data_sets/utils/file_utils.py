import glob
import os


def get_all_filenames(file_name, ext):
    """
    Returns all files under file_name directory with extension ext or file_name if file_name is a file with
    extension ext.
    :param file_name: A file or directory path.
    :param ext: Extension of required files. e.g. '.tif'
    :return: List of files with extension ext.
    """
    if os.path.isdir(file_name):
        return glob.glob(file_name + '/*' + ext)
    elif file_name.endswith(ext):
        return [file_name]
    else:
        return []

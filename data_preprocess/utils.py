import distutils.util
import os
import tarfile
import zipfile


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("------------------------------------------------")


def add_arguments(arg_name, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + arg_name,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


def getfile_insensitive(path):
    """Get the actual file path when given insensitive filename."""
    directory, filename = os.path.split(path)
    directory, filename = (directory or '.'), filename.lower()
    for f in os.listdir(directory):
        newpath = os.path.join(directory, f)
        if os.path.isfile(newpath) and f.lower() == filename:
            return newpath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print(f"Unpacking {filepath} ...")
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar:
        os.remove(filepath)


def unzip(filepath, target_dir):
    """Unzip the file to the target_dir."""
    print(f"Unpacking {filepath} ...")
    tar = zipfile.ZipFile(filepath, 'r')
    tar.extractall(target_dir)
    tar.close()

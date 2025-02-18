import os


def print_separator(label="", char="=", length=100):
    print()
    if label:
        side_length = (length - len(label) - 2) // 2
        print(f"{char * side_length} {label} {char * side_length}".center(length, char))
    else:
        print(char * length)
    print()


def create_folder_if_missing(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

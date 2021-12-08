
import os, shutil

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def filter(target_files):
    filtered = []
    for f in target_files:
        print(f)
        if f.endswith('.c') or f.endswith('.cc') or f.endswith('.cpp'):
            filtered.append(f)
    return filtered

def copy_files(c_only):
    for f in c_only:
        shutil.copy2(f, '/media/nimashiri/DATA/vsprojects/ML_vul_detection/examples/ffmpeg')


def main():
    source_dir = '/media/nimashiri/DATA/vsprojects/FFmpeg-n3.0'
    target_files = getListOfFiles(source_dir)
    c_only = filter(target_files)
    copy_files(c_only)

if __name__ == '__main__':
    main()
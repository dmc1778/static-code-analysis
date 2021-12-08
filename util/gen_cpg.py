
from subprocess import call, check_output, STDOUT, run, check_call
import subprocess

def main():
    source = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/examples/numpy'
    dest = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/cpgs/numpy'
    try:
        check_output(['cd /home/nimashiri/C-Code-Slicer/'], shell=True)
        check_output(['/home/nimashiri/C-Code-Slicer/test-script.sh', 'source', 'dest'], shell=True)
    except subprocess.CalledProcessError as e:
        output = e.output


if __name__ == '__main__':
    main()





import getopt
import os
import sys
from src.fit import fit


def main(argv):
    input_filename = ''
    output_prefix = ''
    model_name = ''
    gpu_id = '-1'
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:g:", ["Input=", "Output=", "Model=", "GPU="])
    except getopt.GetoptError:
        print('usage: main.py -Task <TaskName> --Arguments <ArgumentsList>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -Task <TaskName> --Arguments <ArgumentsList>')
            sys.exit()
        elif opt in ("-i", "--Input"):
            input_filename = arg
        elif opt in ("-o", "--Output"):
            output_prefix = arg
        elif opt in ("-m", "--Model"):
            model_name = arg
        elif opt in ("-g", "--GPU"):
            if arg.isnumeric():
                gpu_id = arg
    if input_filename == '':
        print('usage: main.py -Input <input_filename> --Arguments <ArgumentsList>')
        sys.exit()

    # @TODO. How to check GPUs available, and if the provided number is correct.
    # Otherwise should set the value back to -1.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if os.path.exists(input_filename):
        real_path = os.path.realpath(os.path.dirname(input_filename))
        input_filename = os.path.join(real_path, os.path.basename(input_filename))
    else:
        print('Input filename does not exist on disk, with argument: {}'.format(input_filename))
        sys.exit(2)

    if os.path.exists(os.path.dirname(output_prefix)):
        real_path = os.path.realpath(os.path.dirname(output_prefix))
        output_prefix = os.path.join(real_path, os.path.basename(output_prefix))
    else:
        print('Directory name for the output prefix does not exist on disk, with argument: {}'.format(input_filename))
        sys.exit(2)

    fit(input_filename=input_filename, output_path=output_prefix, selected_model=model_name)


if __name__ == "__main__":
    main(sys.argv[1:])


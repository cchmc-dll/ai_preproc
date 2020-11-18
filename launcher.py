#from  src.run_preprocess import *
from src.run_debug import *

file = '@preproc.args'

print(file)
main(file)


def launch_from_nb(file):
    config_arg = file
    #print(sys.argv[0])
    if config_arg not in sys.argv:
        primary = sys.argv[0]
        sys.argv = [primary,config_arg]
    #print(sys.argv)
    preprocess(file)
    print(file,' processed')

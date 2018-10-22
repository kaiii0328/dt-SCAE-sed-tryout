import argparse
import sys
from custom_parser import add_custom_params

parser = argparse.ArgumentParser(description="Argument Parser for A3LAB Framework")

# Global params
# PATHS
parser.add_argument("-rp", "--root-path", dest="root_path", default='.')
parser.add_argument("-dp", "--dataset-path", dest="dataset_path", default="dataset")

# Configuration file
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)

# Configuration indexing
parser.add_argument("-cid", "--conf-index", dest="confID", default='', type=str,
                    help='id of the configuration used for the experiment')
parser.add_argument("-eid", "--exp-index", dest="expID", default='', type=str,
                    help='sub id of the experiment: if specified the experiment will work in the confID.expID folder.'
                         'Useful if i need to repeat only the post processing phase (in combination with --evaluation-only param ')

# Save on db (still work in progress)
parser.add_argument("-db", "--db-save", dest="db_save", default=False, action="store_true")

# hardware resources
parser.add_argument("-tc", "--tot-core", dest="totCore", default=1, type=int)
parser.add_argument("-gpu", "--use-gpu", dest="use_gpu", default=False, action="store_true")

# fit params
parser.add_argument("-f", "--fit-model", dest="fit_model", default=False, action="store_true")
parser.add_argument("-tob", "--train-on-batch", dest="train_on_batch", default=False, action="store_true")
parser.add_argument("-sv", "--save-model", dest="save_model", default=False, action="store_true")
parser.add_argument("-load", "--load-model", dest="load_model", default=None, type=str)
parser.add_argument("-skp", "--skip-predict", dest="skip_predict", default=False, action="store_true")
parser.add_argument("--evaluation-only", dest="evaluation_only", default=False, action="store_true")

# extend parser with custom parameters
print("Adding custom parameters")
parser = add_custom_params(parser)
args = parser.parse_args(sys.argv[1:])

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        # print('Custom parser line: ' + line)
        if line[0] == '-':
            arguments.extend(line.split())
    # First parse the arguments specified in the config file
    args, unknown = parser.parse_known_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)
    attributes = vars(args)
    # print(attributes)
    for item in sorted(attributes.keys()):
        print('Attribute: {:<30} has value: {:<20}'.format(item, str(attributes[item])))


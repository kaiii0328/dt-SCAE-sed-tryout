from A3LAB_Framework.utility.argument_parser import args
from A3LAB_Framework.utility import utility as utils
import json
import os.path
from shutil import copyfile

def initialization():
    # initialize experiment id string
    STRING_ID = '{:0>4}'.format(args.confID)
    if args.expID:
        STRING_ID = STRING_ID + '.' + args.expID
    else:
        # todo check folder and compute next expID
        pass
    # initialize path tree
    ROOT_PATH, EXP_FOLDER = utils.path_creation(STRING_ID, exp_ID=args.expID, root_path=args.root_path)

    # initialize logger
    utils.initLogger(STRING_ID, EXP_FOLDER)
    # todo implementare salvataggio su db, ora fa solo il dump su json
    if not args.db_save and not args.evaluation_only:
        json_args = json.dumps(args.__dict__)
        config_file_dest_name = os.path.join(EXP_FOLDER, 'args' + STRING_ID)
        with open(os.path.join(config_file_dest_name + '.json'), 'w') as args_file:
            # args_file.write(json.dumps(json_args, indent=4))
            json.dump(json_args, args_file, indent=4)
        copyfile(args.config_filename, config_file_dest_name + '.cfg')

    #elif args.evaluation_only:
    #     # todo se è solo valutazione carico gli argomenti direttamente dal json salvato in modo tale danon dover
    #     # ripassare il config file con i parametri della rete giusti( che nel caso di load-method = weights
    #     # è necessario)
    #     # problema1: questa cosa potrebbe sovrascrivere parte degli args che non volgio.
    #     # possibile soluzone: separare gli agrs della rete da queli dell' esperimento
    #     # problema2 gli altri moduli non la vedono se è fatta cos'
    #     if args.evaluation_only:
    #         args = Dobj()
    #         # load args of the experiment from file
    #         args_json_path = os.path.join(base_paths['experiment_folder'],
    #                                       'args' + base_paths['string_id'] + '.json')
    #         with open(args_json_path) as args_json:
    #             app = json.load(args_json)
    #             args.__dict__ = json.loads(app)

    base_paths = {'root_path': ROOT_PATH, 'experiment_folder': EXP_FOLDER, 'string_id': STRING_ID}

    return base_paths, args

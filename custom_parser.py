import ast
import argparse


class eval_action(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(eval_action, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # print("Custom parser values: " + values)
        values = ast.literal_eval(values)
        # print(values)
        setattr(namespace, self.dest, values)


def add_custom_params(parser):
    # Custom Params
    parser.add_argument("--dataset-list-filepath", dest="dataset_list", default=None, type=str)

    # Evaluation set (DCASE)
    parser.add_argument("--evaluation-list-filepath", dest="evaluation_list", default=None, type=str)
    parser.add_argument("--evaluation-set-path", dest="evaluation_path")
    parser.add_argument("--do-eval", dest="do_evaluation", default=False, action="store_true")
    parser.add_argument("--fit-model-eval", dest="fit_model_eval", default=None, action="store_true")
    parser.add_argument("--load-model-eval", dest="load_model_eval", default=None, action="store_true")

    parser.add_argument('--window-size', dest="window_size", default=256, type=int)

    parser.add_argument("--input-type", dest="input_type", default="logmel")
    parser.add_argument("--file-type", dest="file_type", default="npy")
    parser.add_argument("-kfold", "--kfold-cv", dest="kfold", default=4, type=int)
    parser.add_argument("--standardize-data", dest="standardize_data", default=False, action="store_true")
    parser.add_argument("--split-data", dest="split", default=False, action="store_true")

    parser.add_argument("--test-obj", dest="test_obj", default="baby", choices=["baby", "glass", "gun"])

    # Optimizer params
    parser.add_argument("--callbacks", action=eval_action,
                        default=['TensorBoard', 'CSVLogger', 'LearningRateDecay', 'CustomEarlyStopping'])
    parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta",
                        choices=["adadelta", "adam", "adam-default", "sgd"])
    parser.add_argument("-l", "--loss", dest="loss", default="mse",
                        choices=["mse", "msle", "categorical_crossentropy", "binary_crossentropy", "custom"])
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--batch-size', dest="batch_size", default=100, type=int)
    parser.add_argument('--lr', dest="learning_rate", default=0.001, type=float)
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument("-mu", "--momentum", dest="momentum", default=0.9, type=float)
    parser.add_argument("--early-stopping", dest="early_stopping", default=False, action="store_true")
    parser.add_argument("--shuffle", dest="shuffle", default="True", choices=["True", "False", "batch"])
    parser.add_argument("--validation-split", dest="validation_split", default=0.0, type=float)
    parser.add_argument("--balance-classes", dest="balance_classes", default=False, action="store_true")
    parser.add_argument("--metrics",  default=None, action=eval_action)

    # dropout
    parser.add_argument("--dropout-cnn", dest="dropout_cnn", default=False, action="store_true")
    parser.add_argument("--drop-rate-cnn", dest="drop_rate_cnn", default=0.5, type=float)

    parser.add_argument("--dropout-capsule", dest="dropout_capsule", default=False, action="store_true")
    parser.add_argument("--drop-rate-capsule", dest="drop_rate_capsule", default=0.5, type=float)

    parser.add_argument("--dropout-dense", dest="dropout_dense", default=False, action="store_true")
    parser.add_argument("--drop-rate-dense", dest="drop_rate_dense", default=0.5, type=float)

    # CAPSULE args
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")

    # CAPSULE Layout
    parser.add_argument("--multiple-conv-layers", dest="multiple_conv_layers", default=False, action="store_true")
    #parser.add_argument("--cnn-kernel-number", dest="cnn_kernel_number", default=64, type=int)
    #parser.add_argument("--cnn-kernel-size", dest="cnn_kernel_size", default=3, type=int)
    #parser.add_argument("--cnn-kernel-strides", dest="cnn_kernel_strides", default=1, type=int)
    parser.add_argument("--primary-kernel-size", dest="primary_kernel_size", default=3, type=int)
    parser.add_argument("--primary-kernel-strides", dest="primary_kernel_strides", default=2, type=int)
    parser.add_argument("--primary-n-channels", dest="primary_n_channels", default=16, type=int)
    parser.add_argument("--dim-capsule", dest="dim_capsule", default=16, type=int)
    parser.add_argument("--dim-primary-capsule", dest="dim_primary_capsule", default=8, type=int)

    # Standard CNN LAYOUT
    parser.add_argument("--standard-CNN", dest="standard_CNN", default=False, action="store_true")
    parser.add_argument("-cis", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
    parser.add_argument("-cac", "--cnn-activation", dest="cnn_activation", default="tanh", choices=["linear","tanh", "relu"])
    parser.add_argument("-ci", "--cnn-weight-init", dest="cnn_weight_init", default="glorot_uniform")
    parser.add_argument("-kn", "--kernel-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
    parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action,
                        default=[[3, 3], [3, 3], [3, 3]])
    parser.add_argument("-cs", "--cnn-strides", dest="cnn_strides", action=eval_action,
                        default=[[1, 1], [1, 1], [1, 1]])

    parser.add_argument("--border-mode", dest="border_mode", default="valid", choices=["valid", "same"])
    parser.add_argument("--pooling-border-mode", dest="pooling_border_mode", default="valid", choices=["valid", "same"])
    parser.add_argument("-cwr", "--cnn-w-reg", dest="cnn_w_reg",  default="None")
    parser.add_argument("-cbr", "--cnn-b-reg", dest="cnn_b_reg", default="None")
    parser.add_argument("-car", "--cnn-act-reg", dest="cnn_a_reg", default="None")
    parser.add_argument("-cwc", "--cnn-w-constr", dest="cnn_w_constr", default="None")
    parser.add_argument("-cbc", "--cnn-b-constr", dest="cnn_b_constr", default="None")
    parser.add_argument("-ckc", "--cnn-k-constr", dest="cnn_k_constr", default="None")
    parser.add_argument("--cnn-dilation-rate", dest="dilation_rate", default=1, type=int)
    parser.add_argument("--leaky-relu", dest="leaky_relu", default=False, action="store_true")
    parser.add_argument("--leaky-relu-alpha", dest="leaky_relu_alpha", default=0.3, type=float)


    # Pooling params
    parser.add_argument("-pst", "--pool-strides", dest="pool_strides", action=eval_action, default=[[1, 1], [1, 1], [1, 1]])
    parser.add_argument("-psh", "--pool-shapes", dest="pool_shapes", action=eval_action, default=[[2, 2], [2, 2], [2, 2]])

    # Dense params
    parser.add_argument("-ds", "--dense-layer-shapes", dest="dense_layer_shapes", action=eval_action, default=[64,64])
    parser.add_argument("-dad", "--dense-activation", dest="dense_activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("-dwr", "--d-w-reg", dest="d_w_reg", default="None")
    parser.add_argument("-di", "--dense-weight-init", dest="dense_weight_init", default="glorot_uniform",
                        choices=["glorot_uniform"])

    # bias
    parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action="store_false")

    # batch normalization
    parser.add_argument("-bn", "--batch-norm", dest="batch_norm", default=False, action="store_true")

    # last layer activation

    parser.add_argument("--output-activation", dest="output_activation", default="softmax", choices=["linear", "softmax", "sigmoid"])
    parser.add_argument("--class-labels", dest="class_labels", default='Dcase2017')

    # Evaluation resolution
    parser.add_argument("--eval-resolution", dest="resolution", default=50, type=int) # Number of frames in 1 second

    return parser




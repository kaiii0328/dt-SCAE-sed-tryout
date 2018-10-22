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
    parser.add_argument("--trainset-list-filepath", dest="trainset_list_filepath", default=None, type=str)
    parser.add_argument("--input-type", dest="input_type", default="spectrograms")

    parser.add_argument("-classes", "--class-labels", dest="class_labels", action=eval_action, default=['V', 'O', 'T', 'E'])

    # CNN params
    parser.add_argument("-cis", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
    parser.add_argument("-cac", "--cnn-activation", dest="cnn_activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("-ci", "--cnn-weight-init", dest="cnn_weight_init", default="glorot_uniform")
    parser.add_argument("-kn", "--kernels-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
    parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3, 3], [3, 3], [3, 3]])
    parser.add_argument("-cs", "--cnn-strides", dest="cnn_strides", action=eval_action, default=[[1, 1], [1, 1], [1, 1]])

    parser.add_argument("-bm", "--border-mode", dest="border_mode", default="same", choices=["valid", "same"])
    parser.add_argument("-cwr", "--cnn-w-reg", dest="cnn_w_reg", default="None")
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
    parser.add_argument("-dis", "--dense-input-shape", dest="dense_input_shape", action=eval_action)
    parser.add_argument("-ds", "--dense-layer-shapes", dest="dense_layer_shapes", action=eval_action, default=[64])
    parser.add_argument("-di", "--dense-weight-init", dest="dense_weight_init", default="glorot_uniform",
                    choices=["glorot_uniform"])
    parser.add_argument("-dad", "--dense-activation", dest="dense_activation", default="tanh", choices=["tanh", "relu"])
    # parser.add_argument("-dad", "--dense-activation", dest="dense_activation", action=eval_action, default=["relu", "relu", "relu"])
    parser.add_argument("-dwr", "--d-w-reg", dest="d_w_reg", default="None")
    parser.add_argument("-dbr", "--d-b-reg", dest="d_b_reg", default="None")
    parser.add_argument("-dar", "--d-act-reg", dest="d_a_reg", default="None")
    parser.add_argument("-dwc", "--d-w-constr", dest="d_w_constr", default="None")
    parser.add_argument("-dbc", "--d-b-constr", dest="d_b_constr", default="None")
    parser.add_argument("-dcxt", "--dense-context", dest="dense_context", default=0, type=int)
    parser.add_argument("-pshift", "--predict-shift", dest="pred_shift", default=0, type=int)

    # RNN
    parser.add_argument("-ris", "--rnn-input-shape", dest="rnn_input_shape", action=eval_action)
    parser.add_argument("-rnn", "--rnn-type", dest="rnn_type", default=None, choices=["LSTM", "SimpleRNN", "GRU"])
    parser.add_argument("-rns", "--rnn-layer-shapes", dest="rnn_layer_shapes", action=eval_action, default=[64])
    parser.add_argument("-cxt", "--frame-context", dest="frame_context", default=1, type=int)
    parser.add_argument("-seq", "--return-sequences", dest="return_seq", default=True)
    parser.add_argument("-rac", "--rnn-activation", dest="rnn_activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("-ri", "--rnn-weight-init", dest="rnn_weight_init", default="glorot_uniform",
                        choices=["glorot_uniform"])
    parser.add_argument("-rwr", "--rnn-w-reg", dest="rnn_w_reg", default="None")
    parser.add_argument("-rbr", "--rnn-b-reg", dest="rnn_b_reg", default="None")
    parser.add_argument("-rar", "--rnn-act-reg", dest="rnn_a_reg", default="None")
    parser.add_argument("-rwc", "--rnn-w-constr", dest="rnn_w_constr", default="None")
    parser.add_argument("-rbc", "--rnn-b-constr", dest="rnn_b_constr", default="None")
    parser.add_argument("-bdir", "--bidirectional", dest="bidirectional", default=False, action="store_true")

    # optimizer params
    parser.add_argument("-e", "--epochs", dest="epochs", default=50, type=int)
    parser.add_argument("-ns", "--shuffle", dest="shuffle", default="True", choices=["True", "False", "batch"])
    parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta",
                        choices=["adadelta", "adam", "adam-default", "sgd"])
    parser.add_argument("-l", "--loss", dest="loss", default="mse",
                        choices=["mse", "msle", "categorical_crossentropy", "binary_crossentropy", "custom"])
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=1, type=float)
    parser.add_argument("-mu", "--momentum", dest="momentum", default=0.0, type=float)

    # early stopping params
    parser.add_argument("--early-stopping", dest="early_stopping", default=False, action="store_true")
    parser.add_argument("-vl", "--validation-split", dest="validation_split", default=0.0, type=float)
    parser.add_argument("-pt", "--patience", dest="patience", default=20, type=int)
    parser.add_argument("--monitor-on", dest="monitor_on", default=['val_loss'], choices=['val_loss', 'val_acc'])

    # dropout
    parser.add_argument("-drp", "--dropout", dest="dropout", default=False, action="store_true")
    parser.add_argument("-drpr", "--drop-rate", dest="drop_rate", default=0.5, type=float)

    # bias
    parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action="store_false")

    # batch normalization
    parser.add_argument("-bn", "--batch-norm", dest="batch_norm", default=False, action="store_true")

    return parser




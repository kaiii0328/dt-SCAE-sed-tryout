# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import uniform as ufm, norm
import utility.utility as u

TABS = 4

UNIFORM_STORE = 0

def choice(vet):
    return np.random.choice(vet)


def uniform(interval, type='int', action=None):
    global UNIFORM_STORE
    if type == "int":
        if action == "restore":
            return UNIFORM_STORE

        out = np.random.choice(range(interval[0], interval[1] + 1))
        if action == "store":
            UNIFORM_STORE = out
        return out
    else:
        return ufm.rvs(loc=interval[0], scale=interval[1] - interval[0], size=1)[0]


def loguniform(base=2, low=0, high=1, round_exponent=False, round_output=False):
    exponent = uniform([low, high], 'float')
    if round_exponent:
        exponent = np.round(exponent)
    value = np.power(base, exponent)
    if round_output:
        value = int(np.round(value))
    return value


def normal(loc=0, scale=1, span=None, bounded=False):
    check = False
    while not check:
        value = norm.rvs(loc=loc, scale=scale)
        if span is not None and bounded and span[0] <= value <= span[1]:
            check = True
    return value


def any_val(dist, axis=[1], args=None):
    n = 1
    for a in axis:
        n *= a
    var_list = [0] * n
    for i in range(n):
        var_list[i] = eval(str(dist))

    if axis == [1]:
        if args == "'as_list'":
            return var_list
        else:
            return var_list[0]
    else:
        # print("axis != [1]: " + str(var_list))
        return np.array(var_list).reshape(axis).tolist()


def equal(dist, axis=[1], args=None):
    n = 1
    var = eval(str(dist))
    for a in axis:
        n *= a
    var_list = [var] * n
    if axis == [1]:
        if args == "'as_list'":
            return var_list
        else:
            return var_list[0]
    else:
        return np.array(var_list).reshape(axis).tolist()


def series(dist, axis, directive, args):
    if args is not '':
        guard_value = eval(args)
    else:
        guard_value = None

    n = 1
    for a in axis:
        n *= a
    var_list = [0] * n

    restart = True
    while restart:
        restart = False
        var_list = [0] * n
        var_list[0] = eval(str(dist))
        for i in range(1, n):
            check = False
            while not check:
                var = eval(str(dist))
                if directive == "REPLICATE":    ### direttiva specifica per replicare il drop rate
                    var = var_list[i - 1]
                    check = True
                if directive == "INCREASE":    ### direttiva specifica per incrementare il drop-rate di 0.1
                    var = var_list[i - 1] + 0.1
                    check = True
                if directive == "ENCREASE" and var > var_list[i - 1]:
                    check = True
                if directive == "DECREASE" and var < var_list[i - 1]:
                    check = True
                if directive == "encrease" and var >= var_list[i - 1]:
                    check = True
                if directive == "decrease" and var <= var_list[i - 1]:
                    check = True
            var_list[i] = var
            if guard_value is not None and var == guard_value and i != n - 1:
                restart = True
                break
    return np.array(var_list).reshape(axis).tolist()  # non ha senso per matrici. Da rivedere


def encrease(dist, axis, args):
    return series(dist, axis, 'encrease', args)


def ENCREASE(dist, axis, args):
    return series(dist, axis, 'ENCREASE', args)


def decrease(dist, axis, args):
    return series(dist, axis, 'decrease', args)


def DECREASE(dist, axis, args):
    return series(dist, axis, 'DECREASE', args)


def INCREASE(dist, axis, args):
    return series(dist, axis, 'INCREASE', args)

def REPLICATE(dist, axis, args):
    return series(dist, axis, 'REPLICATE', args)


# ToDo: id esperimento



#   any                 *       qualsiasi valore
#   ENCREASE/DECREASE   >/<     sequenza strettamente crescente/decrescente
#   encrease/decrease   >=/<=   sequenza crescente/decrescente
#   equal               =       unico valore
#   step(x)             +=x     primo valore random poi aumento di x
#   multi(x)            *=x     primo valore random poi moltiplico per x





def randomize(config_file, output_file, start_id=1, config_number=1):
    for id in range(config_number):
        with open(config_file, "r") as f:
            input_lines = f.readlines()
        out_file = open(output_file + '{0:04}'.format(id + start_id) + ".conf", "w")
        out_file.write("--conf-index" + "\t" * TABS + '{0:04}'.format(id + start_id) + "\n")

        alias = {}
        prec_line = ""
        for line in input_lines:
            if prec_line != "":
                line = prec_line + line
            if len(line) > 1 and line.replace(" ", "")[-2] == '\\':
                prec_line = line[:-2]
                continue
            else:
                prec_line = ""

            start_comment = line.find("#")
            line = line[:-1]
            comment = ""
            if start_comment != -1:
                comment = ' ' + line[start_comment:]
                line = line[:start_comment]

            # -----------------------------------------------------------------------------------------------------------------------
            # todo: macro not found
            p_macro = line.find("$")
            while p_macro != -1:
                for k in alias.keys():
                    if line[p_macro + 1:].find(k) == 0:
                        line = line[:p_macro] + str(alias[k]).replace(" ", "") + line[p_macro + len(k) + 1:]
                p_macro = line.find("$")
            # -----------------------------------------------------------------------------------------------------------------------

            start = line.lower().find("rand:")
            if start == -1:
                out_file.write(line + "\n")  # copio la riga tale e quale
                # out_file.write(line + comment + "\n")  # copio la riga tale e quale
            else:
                if line[0:start].replace(" ", "") is not "":
                    out_line = line[0:start].replace(" ", "").replace("\t", "") + " " + "\t" * TABS
                else:
                    out_line = ""
                # find shape directive
                start_shape = line.lower().find("shape{")
                if start_shape == -1:
                    axis = [1]
                else:
                    shape_str = line[start_shape + 6:start_shape + line[start_shape:].find("}")]
                    axis_str = shape_str.replace(" ", "").split(';')

                    axis = [0] * (len(axis_str) - 1)
                    axis_alias = [0] * (len(axis_str) - 1)

                    for i in range(len(axis)):
                        s = axis_str[i].split('=')
                        if len(s) == 2:
                            axis_alias[i] = s[0]
                            s[0] = s[1]
                        axis_str[i] = s[0]

                    if axis_str[-1] is not "":

                        # -----------------------------------------------------------------------------------------------------------------------
                        # valuto l'espressione
                        for a in axis_alias:
                            if a != 0:
                                p = axis_str[-1].find(a)
                                if p != -1:
                                    if p != len(axis_str[-1]) - 1 and (
                                        axis_str[-1][p + 1].isalnum() or axis_str[-1][p + 1] == '_') \
                                            or p != 0 and (axis_str[-1][p - 1].isalnum() or axis_str[-1][p - 1] == '_'):
                                        continue
                                axis_str[-1] = axis_str[-1].replace(a, ";" + a + ";")
                        for i in range(len(axis_alias)):
                            if axis_alias[i] != 0:
                                axis_str[-1] = axis_str[-1].replace(";" + axis_alias[i] + ";", "axis[" + str(i) + "]")

                                # -----------------------------------------------------------------------------------------------------------------------

                    match_condition = False
                    while not match_condition:
                        for i in range(len(axis)):
                            axis[i] = eval(axis_str[i])
                        if axis_str[-1] is not "":
                            if eval(axis_str[-1]):
                                match_condition = True
                        else:
                            match_condition = True

                elements = 1
                for a in axis:
                    elements *= a
                start_vals = line.lower().find("vals{")
                if start_vals == -1:
                    randomized = np.array([0] * elements).reshape(axis).tolist()
                else:
                    val_str = line[start_vals + 5:start_vals + line[start_vals:].find("}")]
                    val_states = val_str.replace(" ", "").split(";")
                    # print('val states: ' + str(val_states))
                    if len(val_states) == 1:
                        val_states.append("any_val")
                    elif val_states[1] == "":
                        val_states[1] = "any_val"

                    args = val_states[1].replace(')', '').split('(')
                    # print('args: ' + str(args))
                    if len(args) == 2:
                        val_states[1] = args[0]
                    else:
                        args.append('')

                    # print('states1: ' + val_states[1] + ' states2: ' + str(val_states[0]) )
                    randomized = eval(val_states[1] + '("' + str(val_states[0]) + '", axis, args[1])')
                    # print(randomized)

                start_alias = line.lower().find("as{")
                if start_alias != -1:
                    key = line[start_alias + 3:start_alias + line[start_alias:].find("}")]
                    alias[key] = randomized

                out_line += str(randomized).replace(" ", "") + "\n"
                # out_line += str(randomized).replace(" ","") + comment + "\n"
                out_file.write(out_line)

    out_file.close()


# per il debug
if __name__ == "__main__":
    config_file = os.path.abspath("./random.conf")
    out_path = os.path.abspath("./config_file/random/")
    u.makedir(out_path)
    randomize(config_file, out_path, start_id=200, config_number=100)

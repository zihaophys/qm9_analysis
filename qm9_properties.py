import os
import qml


def get_qm9_properties(filename):
    """ Returns a dictionary with 12 properties for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    tokens = lines[1].split()
    # xyz_name = 'dsgdb9nsd_000' + str(tokens[1]) + '.xyz'

    property = dict()
    property['mu'] = tokens[5]
    property['alpha'] = tokens[6]
    property['homo'] = tokens[7]
    property['lumo'] = tokens[8]
    property['gap'] = tokens[9]
    property['r2'] = tokens[10]
    property['zpve'] = tokens[11]
    property['U0'] = tokens[12]
    property['U'] = tokens[13]
    property['H'] = tokens[14]
    property['G'] = tokens[15]
    property['Cv'] = tokens[16]

    return property


compounds = [qml.Compound(xyz="qm9/" + f) for f in sorted(os.listdir("qm9/"))]
QM9_Properties = [get_qm9_properties("qm9/" + f) for f in sorted(os.listdir("qm9/"))]

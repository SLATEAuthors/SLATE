import csv
import pickle
import json

def save_json(obj, filename):
    """Save the object as a json file"""
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(filename):
    """Load a json file"""
    with open(filename, 'r') as f:
        obj = json.load(f)

    return obj

def load_pickle(filename):
    """Load a pickled object from the given filename"""
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def save_pickle(obj, filename):
    """Pickle and save the given object to the given file"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def save_list(obj, filename):
    """save the given list to the given file"""
    with open(filename, 'w') as f:
        for item in obj:
            f.write("%s\n" % item)
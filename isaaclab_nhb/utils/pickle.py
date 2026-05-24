import cloudpickle

def p_save(obj,path):
    with open(f"{path}.pickle", "wb") as f:
        cloudpickle.dump(obj, f, protocol=cloudpickle.DEFAULT_PROTOCOL)


def p_load(path):
    with open(f"{path}.pickle", "rb") as f:
        loaded_data = cloudpickle.load(f)
    return loaded_data
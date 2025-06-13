import os
import pickle as pkl

def epoch_from_name(basename) -> int:
    if basename.startswith('model_'):
        return int(basename[12:-4])  # model name is model_epoch_<number>.pkl
    elif basename.startswith('info_'):
        return int(basename[11:-4])  # info name is info_epoch_<number>.pkl
    else:
        raise Exception('does not match known model or info files:', basename)


def get_model_history_srt(model_dir):
    filenames = os.listdir(model_dir)
    if not filenames:
        return ()
    largest = max(map(epoch_from_name, filenames))
    stuff = [[None, None] for _ in range(largest + 1)]
    for fn in filenames:
        idx = epoch_from_name(fn)
        if fn.startswith('model_'):
            stuff[idx][0] = os.path.join(model_dir, fn)
        elif fn.startswith('info_'):
            stuff[idx][1] = os.path.join(model_dir, fn)
    return tuple(tuple(s) for s in stuff if s[0] is not None)


def clear_model_history(save_model_history, model_dir):
    past_models = get_model_history_srt(model_dir=model_dir)
    if save_model_history > -1 and len(past_models) > save_model_history:
        for model_name, info_name in past_models[:len(past_models) - save_model_history]:
            os.remove(model_name)
            os.remove(info_name)


def load_model(MODEL,model_file, info_file,env,):
    model = MODEL.load(model_file, env=env)
    f = open(info_file, 'rb')
    info = pkl.load(f)
    f.close()
    return model, info

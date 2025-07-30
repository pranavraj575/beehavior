"""
method for recovering a dictionary after concatenating all elements into a vector
"""
def deconcater(arr, ksp):
    """
    unconcatenates a concatenated dictionary into a tuple, works with both np arrays and torch tensors,
        and both batched and unbatched array
    Args:
        arr: array output of concatenation of elements of dictionary
        ksp: (keys, shapes, partitions)
            satisifies relation original_dic[keys[i]]=arr[partition[i]:partition[i+1]].reshape(shapes[i])
                keys is ordered keys of dictionary
                partition[i]:partition[i+1] is where keys[i] landed in arr
                keys[i] had shape shapes[i]
    Returns:
        tuple of reshaped elements, in the order of keys
    """
    keys, shapes, partition = ksp
    if len(arr.shape) == 2:
        stuff = [
            arr[:, partition[i]:partition[i + 1]].reshape(arr.shape[0], *shapes[i])
            for i in range(len(keys))
        ]
    else:
        stuff = [
            arr[partition[i]:partition[i + 1]].reshape(shapes[i])
            for i in range(len(keys))
        ]
    return tuple(stuff)

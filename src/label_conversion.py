import json
def find_mapping():
    # This is the file that map the PUG: ImageNet class indexes (from 0 to 151) to the real ImageNet classes (up to 1000)
    with open('/home/pfi/Documents/Data/CVPR_paper/goldeneye/src/class_to_imagenet_idx.json') as f:
        labels = json.load(f)
    labels = dict(sorted(labels.items()))
    # print(labels)
    # Then we create a disctionary that map an imagenet class to the PUG:ImageNet class
    inversed_dict = {}
    counter = 0
    for k,v in labels.items():
        for val in v:
            inversed_dict[int(val)] = counter
        counter = counter + 1
    # print(inversed_dict)
    return inversed_dict

def convert_pred(pred, mapping):
    # print(pred)
    '''
    
    '''
    if pred in mapping.keys():
        return  mapping[pred]
    else:
        label=str(999)+str(pred)
        return int(label) # We put random value if the predicted class is not in the 151 labels we are usingS
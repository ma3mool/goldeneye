from util import *

''' 
Randomizes and returns two lists. Split is between 0-1, and refers to the size of the rank set.
Example, .8 means 80/20 split
'''
def gen_sets(golden_indices, split):
    total = len(golden_indices)
    split_index = int(total*split)
    randomized = random.sample(golden_indices, total)
    rank_set = randomized[0:split_index]
    test_set = randomized[split_index : ]
    return rank_set, test_set

if __name__ == '__main__':
    # Read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

    # Common variables
    range_name = getDNN() + "_" + getDataset()
    range_path = getOutputDir() + "/networkRanges/" + range_name + "/"

    name = getDNN() + "_" + getDataset() + "_real" + getPrecision() + "_sim" + getFormat()
    if getQuantize_en(): name += "_" + "quant"

    netProfilePath = getOutputDir() + "/networkProfiles/" + name + "/"
    outPath = getOutputDir() + "/data_subset/" + name + "/"
    golden_data = load_file(netProfilePath + "golden_data")

    split_ratio = .8
    
    # generate an Analysis Set (AS) and Deployment Set (DS)
    if "IMAGENET" in getDataset():  images_base = list(range(0,50000))
    elif "CIFAR" in getDataset():   images_base = list(range(0,10000))
    
    random.seed(9001)
    analysis_set, deployment_set= gen_sets(images_base, split_ratio)
    save_data(outPath, "analysis_set", analysis_set)
    save_data(outPath, "deployment_set", deployment_set)
    random.seed()  #back to randomness

    # generate a list from the correct images in AS and DS
    ASgoodImgs = []
    DSgoodImgs = []
    for i in analysis_set:
        if golden_data[i][0]:
            ASgoodImgs.append(i)

    for i in deployment_set:
        if golden_data[i][0]:
            DSgoodImgs.append(i)

    save_data(outPath, "rank_set_good", ASgoodImgs)
    save_data(outPath, "test_set_good", DSgoodImgs)

    # CSVs of imgs
    f = open(outPath + "AS.csv", "w+")
    for i in range(len(analysis_set)):
        outputString = "%d\n" %(analysis_set[i])
        f.write(outputString)
    f.close()

    f = open(outPath + "DS.csv", "w+")
    for i in range(len(deployment_set)):
        outputString = "%d\n" %(deployment_set[i])
        f.write(outputString)
    f.close()

    f = open(outPath + "AS_good.csv", "w+")
    for i in range(len(ASgoodImgs)):
        outputString = "%d\n" %(ASgoodImgs[i])
        f.write(outputString)
    f.close()

    f = open(outPath + "DS_good.csv", "w+")
    for i in range(len(DSgoodImgs)):
        outputString = "%d\n" %(DSgoodImgs[i])
        f.write(outputString)
    f.close()

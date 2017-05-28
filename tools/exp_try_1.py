def RF_calculation(conv, layerIn):
    # RF_calculation(nb_features, kernerl, padding, strides, receptive_file=1, jump=1, start_point=0.5):
    nb_features = layerIn[0]
    jump = layerIn[1]
    receptive_file = layerIn[2]
    start_point = layerIn[3]

    kernerl = conv[0]
    strides = conv[1]
    padding = conv[2]


    nb_features_nest = int((nb_features + 2*padding -kernerl)/strides)+1
    jump_next = jump*strides
    receptive_file_next = receptive_file + (kernerl - 1) * jump
    start_point_next = start_point + ((kernerl - 1) / 2 - padding) * jump

    return nb_features_nest, jump_next, receptive_file_next, start_point_next

def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \n \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))


layerInfos = []

if __name__ == '__main__':

    convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
    layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']


    imsize = 227
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")

    for i in range(len(convnet)):
        currentLayer = RF_calculation(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])


    print ("--------------------------------------------------------------------")
    layer_name = raw_input("Layer name where the feature in: ")
    layer_idx = layer_names.index(layer_name)
    idx_x = int(raw_input("index of the feature in x dimension (from 0)"))
    idx_y = int(raw_input("index of the feature in y dimension (from 0)"))

    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert (idx_x < n)
    assert (idx_y < n)

    print ("receptive field: (%s, %s)" % (r, r))
    print ("center: (%s, %s)" % (start + idx_x * j, start + idx_y * j))


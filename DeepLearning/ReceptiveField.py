# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k: kernel size
# - s: stride
# - p: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
#Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math

nets = {'alexnet': {'layers': [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0], [6, 1, 0], [1, 1, 0]],
                          'name': ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']},
        'vgg16': {'layers': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                        [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0]],
                'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                            'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5']},
        'zf-5': {'layers': [[7, 2, 3], [3, 2, 1], [5, 2, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}}

def out_from_in(layer, IN):
    '''@layer : kernel parameters [kernel_size, stride, padding]
       @IN : [feature_size, jump_size, receptive_field, first_feature_position]
    '''
    n_in, j_in, r_in, start_in = IN
    k, s, p = layer

    n_out = math.floor((n_in + 2 * p - k) / s) + 1
    actual_P = (n_out - 1) * s - n_in + k
    pR = math.ceil(actual_P / 2)
    pL = math.floor(actual_P / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) // 2 - pL) * j_in

    return (n_out, j_out, r_out, start_out)


def printLayer(OUT, layer_name):
    print(layer_name + ":")
    print("\t n features: %d\t jump: %d\t receptive size: %d\t start: %.2f " % OUT)
 

if __name__ == "__main__":
    net_name = 'alexnet'
    net_name = net_name.lower()
    layers = nets[net_name]['layers']
    layer_names = nets[net_name]['name']
    curIN = (227, 1, 1, 0.5)
    print("-------Net summary------")
    printLayer(curIN, 'input')
    for idx, layer in enumerate(layers):
        OUT = out_from_in(layer, curIN)
        printLayer(OUT, layer_names[idx])
        curIN = OUT
    print("------------------------")

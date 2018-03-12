import scipy.io as sio

def read_mat_data():
    mat = sio.loadmat('powerTrainData.mat')
    trainInput=mat['powerTrainInput']
    trainDate=mat['powerTrainDate']
    trainOutput=mat['powerTrainOutput']
    testInput=mat['powerTestInput']
    return trainInput, trainDate, trainOutput, testInput

from raw_data import read_mat_data
import matplotlib.pyplot as plt

trainInput, trainDate, trainOutput, testInput = read_mat_data()

#trainpower=trainInput[0,:]
plt.plot(trainDate, trainOutput)

plt.show()
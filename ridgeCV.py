from raw_data import read_mat_data
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


trainInput, trainDate, trainOutput, testInput = read_mat_data()
x_train, x_test, y_train, y_test = train_test_split(trainInput, trainOutput, test_size=0.3)

mdl = sklm.RidgeCV()

reg = mdl.fit(x_train,y_train)
pred = mdl.predict(x_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test,pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,pred))
print('Standard Deviation: %.2f' % np.std(pred))

# Plot outputs
fig, ax = plt.subplots(dpi=150)
ax.scatter(y_test,pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3,color='red')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Ridge Regression with CV Model Training')

testOutput=mdl.predict(testInput)

fig2, ax2 = plt.subplots(dpi=150)
ax2.plot(testOutput)
ax2.set_ylabel('Power (MW)')
ax2.set_title('Load Forecasting with Ridge Regression Model')
plt.show()

# Write predictions to a text file
np.savetxt('ridgeCV_pred.txt', testOutput, fmt='%.f')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import keras.backend as K

all_xs = []
all_ys = []
filename = "poyntingdata.csv"
filename2 = "testdata.csv"

with open(filename) as inf:
        for line in inf:
                poynting_coefficient, t1, t2 = line.strip().split(",")
                poynting_coefficient = float(poynting_coefficient)
                t1 = float(t1)
                t2 = float(t2)
                linelist = [t1, t2]
                all_xs.append(linelist)
                all_ys.append(poynting_coefficient)

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])
model = Sequential()
model.add(Dropout(0.2, input_shape=(2,)))
model.add(Dense(16, input_shape=(2,)))
model.add(Dense(16, input_shape=(2,)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit(all_xs, all_ys, epochs=50, verbose=1)
test_xs = []
test_ys = []

with open(filename2) as inf:
        for line in inf:
                poynting_co, x1, x2 = line.strip().split(",")
                poynting_co = float(poynting_co)
                x1 = float(x1)
                x2 = float(x2)
                testlist = [x1, x2]
                test_xs.append(testlist)
                test_ys.append(poynting_co)

test_xs = np.array(test_xs)
test_ys = np.transpose([test_ys])
error = model.evaluate(test_xs, test_ys, verbose=1)  
print(error) 
midpred = model.predict(test_xs, verbose=1)
np.savetxt("firststep.csv", midpred, delimiter=",")
model.trainable = False
model1 = Sequential()
model1.add(Dropout(0.2, input_shape=(1,)))
model1.add(Dense(16, input_shape=(1,)))
model1.add(Dense(16, input_shape=(1,)))
model1.add(Dense(2))
model1.add(model)
model1.compile(loss='mean_squared_error', optimizer='Adam')
model1.fit(all_ys, all_ys, epochs=50, verbose=1)
toterror = model1.evaluate(test_ys, test_ys, verbose=1)
print(toterror)
pred = model1.predict(test_ys, verbose=1)
np.savetxt("tandemres.csv", pred[:,0], delimiter=",")

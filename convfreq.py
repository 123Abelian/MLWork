import S4
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, InputLayer, Input, Reshape, MaxPooling2D, UpSampling2D
import numpy as np
from pprint import pprint

all_xs = []
all_ys = []
test_xs = []
test_ys = []

arraylist = []
coefflist = []
for x1 in np.arange(0.1, 1.3, 0.01):
        for x2 in np.arange(0.1, 1.3, 0.01):
                S = S4.New(Lattice = ((1, 0), (0, 1)), NumBasis=5)
                S.SetMaterial(Name = "Ramanium1", Epsilon = -100)
                S.SetMaterial(Name = "Ramanium2", Epsilon = 12.25)
                S.SetMaterial(Name = "Ramanium3", Epsilon = 4)
                S.SetMaterial(Name = "Ramanium4", Epsilon = 2)
                S.SetMaterial(Name = "Vacuum", Epsilon = 1)
                S.AddLayer(Name = 'AirAbove', Thickness = 0, S4_Material = "Vacuum")
                S.AddLayer(Name = 'Slab', Thickness = x1, S4_Material = "Ramanium4")
                S.AddLayer(Name = 'Slab2', Thickness = x2, S4_Material = "Ramanium3")
                S.AddLayer(Name = 'Slab3', Thickness = 0.5, S4_Material = "Ramanium2")
                S.AddLayer(Name = 'Slab4', Thickness = 0.5, S4_Material = "Ramanium1")
                S.AddLayerCopy(Name = 'AirBelow', Thickness = 0, S4_Layer = "AirAbove")
                S.SetExcitationPlanewave(IncidenceAngles = (0, 0),
 sAmplitude = 0.707, pAmplitude = 0.707, Order = 0)
		
		pcarray = np.zeros(10, dtype='complex')
		i = 0
		for f1 in np.arange(0.5, 1.0, 0.05):
                	S.SetFrequency(f1)
                	forward, backward = S.GetPoyntingFlux('AirAbove', 0)
                	t = backward/forward
			pcarray[i] = t
			i += 1
		
		pprint(pcarray)
		print(pcarray.shape)
		coefflist.append(pcarray)
			
		eparray = np.zeros((50, 50), dtype='complex')
		for y1 in range(0, 50):
			for y2 in range(0, 50):
				eparray[y1][y2] = S.GetEpsilon(y1, 0, (x1 + x2 + 1)/50 * y2)
		
		pprint(eparray)
		print(eparray.shape)
		arraylist.append(eparray)

all_xs = np.dstack(arraylist)
all_ys = np.dstack(coefflist)
all_ys = np.transpose([all_ys])
all_ys = np.squeeze(all_ys)
all_xs = np.transpose([all_xs])
print(all_xs.shape)


cb = keras.callbacks.TensorBoard(log_dir='/tmp/conv1', histogram_freq=0, write_graph=True)
model = Sequential()
model.add(Conv2D(64, 1, activation='relu'))
model.add(MaxPooling2D(pool_size=1, strides=2))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(16))
model.add(Dense(10))
model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit(all_xs, all_ys, epochs=1000, batch_size = 1000, callbacks=[cb])
model.trainable = False

arraylist1 = []
coefflist1 = []
for z1 in np.arange(0.4, 3.3, 0.01):
        for z2 in np.arange(0.4, 3.3, 0.01):
                S1 = S4.New(Lattice = ((1, 0), (0, 1)), NumBasis=5)
                S1.SetMaterial(Name = "Ramanium1", Epsilon = -100 + 0.5j)
                S1.SetMaterial(Name = "Ramanium2", Epsilon = 12.25 + 0.5j)
                S1.SetMaterial(Name = "Ramanium3", Epsilon = 4)
                S1.SetMaterial(Name = "Ramanium4", Epsilon = 2)
                S1.SetMaterial(Name = "Vacuum", Epsilon = 1)
                S1.AddLayer(Name = 'AirAbove', Thickness = 0, S4_Material = "Vacuum")
                S1.AddLayer(Name = 'Slab', Thickness = x1, S4_Material = "Ramanium4")
                S1.AddLayer(Name = 'Slab2', Thickness = x2, S4_Material = "Ramanium3")
                S1.AddLayer(Name = 'Slab3', Thickness = 0.5, S4_Material = "Ramanium2")
                S1.AddLayer(Name = 'Slab4', Thickness = 0.5, S4_Material = "Ramanium1")
                S1.AddLayerCopy(Name = 'AirBelow', Thickness = 0, S4_Layer = "AirAbove")
                S1.SetExcitationPlanewave(IncidenceAngles = (0, 0),
 sAmplitude = 0.707, pAmplitude = 0.707, Order = 0)

		pcarray1 = np.zeros(10, dtype='complex')
                i1 = 0
                for f2 in np.arange(0.5, 1.0, 0.05):
                        S1.SetFrequency(f2)
                        forward1, backward1 = S1.GetPoyntingFlux('AirAbove', 0)
                        t1 = backward1/forward1
                        pcarray1[i1] = t1
			i1 += 1
		
		pprint(pcarray1)
                print(pcarray1.shape)
                coefflist1.append(pcarray1)
		
                eparray1 = np.zeros((50, 50), dtype=complex)
                for a1 in range(0, 50):
                        for a2 in range(0, 50):
                                eparray1[a1][a2] = S.GetEpsilon(a1, 0, (z1 + z2 + 1)/50 * a2)

                pprint(eparray1)
                print(eparray1.shape)
                arraylist1.append(eparray1)

test_xs = np.dstack(arraylist1)
test_ys = np.dstack(coefflist1)
test_ys = np.squeeze(test_ys)
test_ys = np.transpose([test_ys])
test_ys = np.squeeze(test_ys)
test_xs = np.transpose([test_xs])
print(test_xs.shape)
testloss = model.evaluate(test_xs, test_ys)
print(testloss)
pred = model.predict(test_xs, verbose=1)
np.savetxt("groundtruthres.csv", test_ys, delimiter=",")
np.savetxt("convres.csv", pred, delimiter=",")

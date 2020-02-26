import tensorflow as tf
import numpy as np
import math

def trainAndTest(train_images, train_labels, test_images):
    x0=train_images[:,0]*0+1
    x1=train_images[:,0]
    x2=train_images[:,0]*train_images[:,0]
    x3=train_images[:,0]*train_images[:,0]*train_images[:,0]
    x4=train_images[:,0]*train_images[:,0]*train_images[:,0]*train_images[:,0]
    x5=train_images[:,1]
    x=np.array([x0, x1, x2, x3, x4, x5])
    A=np.dot(x, np.transpose(x))+0.000001*np.eye(6)
    b=np.dot(x, train_labels)
    coeff=np.linalg.solve(A, b) 
    xx0=test_images[:,0]*0+1
    xx1=test_images[:,0]
    xx2=test_images[:,0]*test_images[:,0]
    xx3=test_images[:,0]*test_images[:,0]*test_images[:,0]
    xx4=test_images[:,0]*test_images[:,0]*test_images[:,0]*test_images[:,0]
    xx5=test_images[:,1]
    xxx=np.array([xx0, xx1, xx2, xx3, xx4, xx5])
    return np.dot(np.transpose(xxx), np.transpose(coeff))

def payoff(xxx, nnn, r, strike, dt):
    xmaxs=np.amax(xxx, axis=0) - strike
    xmaxsm0=np.amax(np.array([xmaxs, xmaxs * 0]), axis=0)
    return math.exp(-dt * nnn * r) * xmaxsm0

def BermudanCall1D(spot, strike, sigma, delta, r, N, T, M):
    N=int(N)
    dt=T/(N-1)
    prop=0.5
    nums=int(prop*M)
    paths=np.zeros(M*N*1).reshape(N,1,M)
    paths[0,:,:]=np.zeros(M*1).reshape(1,M)+spot
    values=np.zeros(M*N).reshape(N,M)

    for i in np.arange(1, N):
        ran=np.random.normal(0, 1, int(M/2)).reshape(1, int(M/2))  
        ran=np.concatenate((ran,ran*-1.0),axis=1)
        paths[i,:,:]= paths[i-1,:,:]*np.exp((r - delta - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * ran)

    contImp = payoff(paths[N-1,:,:], N-1, r, strike, dt)

    for i in np.arange((N-2), -1, -1):
        payoffs=payoff(paths[i, :, :], i, r, strike, dt)       
        xx=np.concatenate((paths[i, :, np.arange(0, M)], payoffs[np.arange(0,M)].reshape(M,1)),axis=1)
        x_train = xx[np.arange(0, nums),:]
        y_train=contImp[np.arange(0,nums)]   
        cont=0
        if i==0:
            cont=cont*0+np.sum(contImp)/M       
        else:
            cont=trainAndTest(x_train, y_train, xx)  
        contImp = (payoffs>cont)*payoffs + (payoffs<=cont)*contImp
    return np.sum(contImp)/M

R=1000

results1D=np.zeros(R*5).reshape(R,5)

for i in np.arange(R):
        spot=1.0
        strike=1.0*math.exp(np.random.normal(0, 1))
        sigma=0.1*math.exp(np.random.normal(0, 1))
        delta=0.1*math.exp(np.random.normal(0, 1))
        r=0.05*math.exp(0.2*np.random.normal(0, 1))
        N=10
        T=1.0
        M=int(1e4)
        res=BermudanCall1D(spot, strike, sigma, delta, r, N, T, M)
        lin=np.array([strike, sigma, delta, r, res])
        results1D[i,:]=lin

model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(1)
    ])   
 
model.compile(loss=tf.keras.losses.mean_squared_error) 
 
model.fit(results1D[:, np.arange(0, 4)], results1D[:,4] / 10000.0, epochs=10)

def BermudanCall1DNetwork(spot, strike, sigma, delta, r, N, T):   
    newparam= np.array([strike/spot, sigma*math.sqrt(T), delta*T, r*T]).reshape(1, 4)
    #return BermudanCall1D(1.0, strike/spot,sigma*math.sqrt(T), delta*T, r*T, N, 1, int(1e6))*spot 
    return model.predict(newparam)*10000.0*spot   

print(BermudanCall1DNetwork(90,100,0.2,0.1,0.05,10,3.0))
print(BermudanCall1DNetwork(1.0,100.0/90.0,0.2*math.sqrt(3.0),0.1*3.0,0.05*3.0,10,1.0)*90.0)
print(BermudanCall1D(90,100,0.2,0.1,0.05,10,3.0,int(1e6)))
print(BermudanCall1D(90,100,0.2*math.sqrt(20.0),0.1*20.0,0.05*20.0,10,3.0/20.0,int(1e6)))
print(BermudanCall1D(1,100.0/90.0,0.2,0.1,0.05,10,3.0,int(1e6))*90.0)
print(BermudanCall1D(1,100.0/90.0,0.2*math.sqrt(3.0),0.1*3.0,0.05*3.0,10,1.0,int(1e6))*90.0) 

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import exp
import matplotlib.pyplot as plt
import pickle
import logging
from copy import deepcopy
from time import strftime, localtime
from datetime import timedelta
from datetime import datetime


global retaincolumn
retaincolumn=[]

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
#implementing sigmoid formula
def sigmoid(z):
    sig = 1 / (1 + exp(-z))
    return sig
   
def toOnHotVector(train_label_vec):
    noOfData = len(train_label_vec)
    data1hot = np.zeros(shape = (noOfData,2))#dummy matrix with all zeroes0
    for i in range(noOfData):
            col = np.int(train_label_vec[i])
            data1hot[i,col] = 1.0
    return data1hot

def feed_forward(data_i,w1,w2):
    sigmoid_vector=np.vectorize(sigmoid)
    w1_transpose=np.transpose(w1)
    #print("w1 ",w1.shape)
    netj_hidden_all=np.dot(data_i,w1_transpose)
    #print("data_i ",data_i.shape)
    
    #print("w1_transpose ",w1_transpose.shape)
    #print("netj_hidden_all ",netj_hidden_all.shape)
    zj_all_withoutbias=sigmoid_vector(netj_hidden_all)
    #print("zj_all_withoutbias ",zj_all_withoutbias.shape)
    
    bias=np.double(np.ones(zj_all_withoutbias.shape[0]))
    #print("bias ",bias.shape)
    zj_withbias=np.column_stack((zj_all_withoutbias,bias))
    #print("zj_withbias ",zj_withbias.shape)
    w2_transpose=np.transpose(w2)
    netj_output_all=np.dot(zj_withbias,w2_transpose)
    #print("zj_withbias ",zj_withbias.shape)
    #print("w2_transpose ",w2_transpose.shape)
    #print("netj_output_all ",netj_output_all.shape)
    ol_all=sigmoid_vector(netj_output_all)
    #print("ol_all ",ol_all.shape)
    return zj_withbias,zj_all_withoutbias,ol_all
     
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
#-------------------Batch GD----------------------------------------------------  
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    #print("w1 nnobj ", w1.shape)
    #print("w2 nnobj ", w2.shape)
    
    obj_val = 0
    obj_grad = np.array([])
    train_label_1tok = []
    train_label_1tok = toOnHotVector(train_label)#converting label to 1hot vector
    yil_all=train_label_1tok
    bias = np.double(np.ones(train_data.shape[0]))
    train_data_withbias = np.column_stack((train_data,bias))#Add bias to all the training data
    J = 0.0 # contains output error for all data**  
    gradient1 = 0.0
    gradient2 = 0.0
    zj_all_withbias,zj_all_withoutbias,oil_all=feed_forward(train_data_withbias,w1,w2)
    #print("oil_all-outer ",oil_all[1,])   
    delta = oil_all - yil_all    
    #print("delta-outer ",delta[1,]) 
    ln_oil=np.log(oil_all)  
    ln_1_sub_oil=np.log(1-oil_all)
    #print("yil_all ",yil_all.shape)
    #print("ln_oil ",ln_oil.shape)
    #print("ln_1_sub_oil ",ln_1_sub_oil.shape)
    Jil=(np.multiply(yil_all,ln_oil)) + np.multiply((1-yil_all),ln_1_sub_oil)#reccommended by ashi
    #-------------summing by part-------------
    Ji=np.sum(Jil,axis=1)*(-1)#equation 7
    #print("Ji ",Ji.shape)
    J=np.sum(Ji,axis=0)/train_data.shape[0]#equation 6
    #print("Ji ",Ji.shape)
    #-------------------
    #------------------sum at once
#    sum = np.sum(Jil)
#
#    divisor=(-1)*len(train_data)
#
#    J=sum/divisor
    #---------------
    delta_transpose=np.transpose(delta) 
    gradient2=np.dot(delta_transpose,zj_all_withbias)
    #print("delta_transpose ",delta_transpose.shape)
    #print("gradient2 ",gradient2.shape)
    #print("zj_all_withbias ",zj_all_withbias.shape)

    #print("w2",w2.shape)
    #print("gradient2 inner 1=   ",gradient2.shape)
    gradient2 = (gradient2 + (lambdaval * w2))/train_data.shape[0] #equation 16
    #print("gradient2 inner 2=   ",gradient2.shape)

    w2=w2[:,:-1] 
    delta_mult_wlj=np.dot(delta,w2)
    term1=(1-zj_all_withoutbias)*zj_all_withoutbias
    term2=term1*delta_mult_wlj
    term2_transpose=np.transpose(term2)
    gradient1 = np.dot(term2_transpose,train_data_withbias)
    gradient1 = (gradient1 + (lambdaval * w1))/train_data.shape[0] #equation 12
    
    
    #obj_val calculation 
    #first term
    
    #------------Approach1
    #sumD=np.sum(w1**2,axis=1)
    #sumM=np.sum(sumD,axis=0)
    ##second term
    #sumM2=np.sum(w2**2,axis=1)
    #sumK=np.sum(sumM2,axis=0)
    #
    #J+=(lambdaval*(sumM+sumK))/(2*train_data_withbias.shape[0])
    #obj_val=J
    
    #---------------Ends here-------
    #obj_val=J
    
    #print("gradient2 outer=   ",gradient2.shape)
    #print("gradient2 outer=   ",gradient2.shape)

    #obj_grad calculation
    obj_grad = np.concatenate((gradient1.flatten(),gradient2.flatten()),0)
    
    
    #---------different approach
    #Regularization Start

    

    reg_func_param2 = np.sum(w1*w1)

    reg_func_param3 = np.sum(w2*w2)    

    

    reg_func_param4 = (lambdaval/(2*len(train_data)))*(reg_func_param2 + reg_func_param3)

    reg_func = J + reg_func_param4
    obj_val=reg_func
    #-----------------ends here----------
    print("obj_val ",obj_val)
    return (obj_val,obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    bias = np.double(np.ones(data.shape[0]))
    data = np.column_stack((data,bias))
    labels = []
    zj_withbias,zj_all_withoutbias,ol_all = feed_forward(data,w1,w2)
    labels=np.argmax(ol_all,axis=1)
    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('facenn.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# set the number of nodes in input unit (not including bias unit)

n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

#n_hidden_range = (4,21,4)
#lambda_range = (0,61,5)
lambdaval=10
#for n_hidden in np.arange(*n_hidden_range):
#    for lambdaval in np.arange(*lambda_range):
        # initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.


start = datetime.now() 
shh=start.hour
smm=start.minute
sss=start.second  
start_time=str(shh)+":"+str(smm)+":"+str(sss)    
#print(start_time)

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')

end = datetime.now() 
ehh=end.hour
emm=end.minute
ess=end.second
end_time=str(ehh)+":"+str(emm)+":"+str(ess)
#print(end_time)
FMT = '%H:%M:%S'
tdelta = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
difference= str(tdelta)
#print('difference',difference)
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))        
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
train_acc = str(100*np.mean((predicted_label == train_label).astype(float)))
print('\n Training set Accuracy:' + str(train_acc) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
valid_acc = str(100*np.mean((predicted_label == validation_label).astype(float)))
print('\n Validation set Accuracy:'+ str(valid_acc) + '%')
    
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
test_acc = str(100*np.mean((predicted_label == test_label).astype(float)))
print('\n Test set Accuracy:' + str(test_acc) + '%')

hiddenStr = str(n_hidden)
lambdaStr = str(lambdaval)
#logging.info('%s',hiddenStr)
logger.info(hiddenStr + '\t' + lambdaStr+ '\t' + str(train_acc)+ '\t' + str(valid_acc)+ '\t' + str(test_acc)+'\t'+difference)

#obj = [retaincolumn, n_hidden, w1, w2, lambdaval]
pickle_name=str(n_hidden)+"hidden_"+str(lambdaval)+"lambda.pickle"
pickle.dump((n_hidden,w1,w2,lambdaval),open(pickle_name,"wb"))

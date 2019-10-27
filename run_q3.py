import numpy as np
import scipy.io
from nn import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sn
import pandas as pd

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 52
# pick a batch size, learning rate
batch_size = 32
learning_rate = 8e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,64,params,'hidden')
initialize_weights(64,64,params,'hidden2')
initialize_weights(64,36,params,'output')

# fig = plt.figure(1, (8, 7))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                  axes_pad=0.2,  # pad between axes in inch.
#                  )
# for i in range(64):
#     grid[i].imshow(params['W' + 'layer1'][:,i].reshape(32,32))  # The AxesGrid object work as a list of axes.
# plt.title("Weights Right After Initialization")
# plt.show()


assert(params['Wlayer1'].shape == (1024,64))
assert(params['blayer1'].shape == (64,))
assert(params['Whidden'].shape == (64,64))
assert(params['bhidden'].shape == (64,))
assert(params['Woutput'].shape == (64,36))
assert(params['boutput'].shape == (36,))

h1 = forward(train_x, params,'layer1')
h2 = forward(h1, params,'hidden')
h3 = forward(h2, params,'hidden2')
probs = forward(h3,params,'output',softmax)
print("h1 shape: " + str(h1.shape))
print("h2 shape: " + str(h2.shape))
print("output shape: " + str(probs.shape))


loss, acc = compute_loss_and_acc(train_y, probs)
print("{}, {:.2f}".format(loss,acc))

delta1 = probs

delta2 = backwards(delta1,params,'output',linear_deriv)
delta3 = backwards(delta2,params,'hidden2',sigmoid_deriv)
delta4 = backwards(delta3,params,'hidden',sigmoid_deriv)
backwards(delta4,params,'layer1',sigmoid_deriv)



trainAcc = np.zeros(max_iters)
validAcc = np.zeros(max_iters)
avg_loss = np.zeros(max_iters)

confusion_matrix_valid = np.zeros((train_y.shape[1],train_y.shape[1]))
confusion_matrix_train = np.zeros((train_y.shape[1],train_y.shape[1]))

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    batches = get_random_batches(train_x, train_y, batch_size)
    for xb,yb in batches:
        pass
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb,params,'layer1')
        h2 = forward(h1, params, 'hidden')
        h3 = forward(h2, params, 'hidden2')
        probs = forward(h3,params,'output',softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        confusion_matrix_train[np.argmax(yb, axis = 1), np.argmax(probs, axis = 1)] +=1

        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs
        delta1 -= yb


        delta2 = backwards(delta1,params,'output',linear_deriv)
        delta3 = backwards(delta2, params, 'hidden2',sigmoid_deriv)
        delta4 = backwards(delta3, params, 'hidden', sigmoid_deriv)
        backwards(delta4,params,'layer1',sigmoid_deriv)


        # apply gradient
        params['W'+'output'] -= learning_rate * params['grad_W' + 'output']
        params['b'+'output'] -= learning_rate * params['grad_b' + 'output']
        params['W'+'hidden2'] -= learning_rate * params['grad_W' + 'hidden2']
        params['b'+'hidden2'] -= learning_rate * params['grad_b' + 'hidden2']
        params['W'+'hidden'] -= learning_rate * params['grad_W' + 'hidden']
        params['b'+'hidden'] -= learning_rate * params['grad_b' + 'hidden']
        params['W'+'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b'+'layer1'] -= learning_rate * params['grad_b' + 'layer1']


    total_acc/=len(batches)


    #------- validation accuracy --------#
    h1Valid = forward(valid_x,params,'layer1')
    h2Valid = forward(h1Valid, params, 'hidden')
    h3Valid = forward(h2Valid, params, 'hidden2')
    probs = forward(h3Valid, params, 'output', softmax)
    lossValid, valid_acc = compute_loss_and_acc(valid_y, probs)

    confusion_matrix_valid[np.argmax(probs, axis = 1), np.argmax(valid_y, axis = 1)] +=1
    
    validAcc[itr] = valid_acc
    trainAcc[itr] = total_acc

    avg_loss[itr] = total_loss/len(train_x)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x,params,'layer1')
h2 = forward(h1, params, 'hidden')
h3 = forward(h2, params, 'hidden2')
probs = forward(h3, params, 'output', softmax)
loss, trained_acc = compute_loss_and_acc(valid_y, probs)


print('Validation accuracy: ',trained_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# epochs = np.arange(max_iters)
# #create the figure
# fig = plt.figure()
# ax = plt.axes()
# ax.grid()
# ax.set_title("Accuracy over epochs")

# ax.plot(epochs, trainAcc)
# ax.plot(epochs, validAcc)
# #plt.legend(title = "Accuracy legend", title_fontsize = 12, handles=legend, bbox_to_anchor=(1.33, 0.6))
# plt.legend(('Training accuracy', 'Validation accuracy'))
# plt.show()
#plt.close()

fig1 = plt.figure()
ax1 = plt.axes()
ax1.grid()
ax1.set_title("Avg Loss over epochs")

ax1.plot(epochs, avg_loss)
#plt.legend(title = "Accuracy legend", title_fontsize = 12, handles=legend, bbox_to_anchor=(1.33, 0.6))
fig1.legend(('CE Loss'))
plt.show()


# Q3.1.3
fig = plt.figure(1, (8, 7))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
for i in range(64):
    grid[i].imshow(params['W' + 'layer1'][:,i].reshape(32,32))  # The AxesGrid object work as a list of axes.
plt.title("Weights After Being Trained")
plt.show()




# Q3.1.4
#confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# import string
# plt.imshow(confusion_matrix_valid,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.title("Validation Image Confusion Matrix")
# plt.show()

# plt.imshow(confusion_matrix_train,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.title("Training Image Confusion Matrix")
# plt.show()
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale*np.random.randn(int(num_filters*(H/2)*(W/2)),hidden_dim) # Pooling reduces the dimension of H and W
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_relu_pool_out,conv_relu_pool_cache = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    affine_relu_out,affine_relu_cache = affine_relu_forward(conv_relu_pool_out,W2,b2)
    affine_out,affine_cache = affine_forward(affine_relu_out,W3,b3)
    scores = affine_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscores = softmax_loss(scores,y)

    daffine_relu_out,dW3_raw,db3_raw = affine_backward(dscores,affine_cache)
    loss += 0.5*self.reg*np.sum(self.params['W3']*self.params['W3'])
    grads['W3'] = dW3_raw+self.reg*self.params['W3']
    grads['b3'] = db3_raw

    dconv_relu_pool_out,dW2_raw,db2_raw = affine_relu_backward(daffine_relu_out,affine_relu_cache)
    loss += 0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
    grads['W2'] = dW2_raw+self.reg*self.params['W2']
    grads['b2'] = db2_raw

    dX,dW1_raw,db1_raw = conv_relu_pool_backward(dconv_relu_pool_out,conv_relu_pool_cache)
    loss += 0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])
    grads['W1'] = dW1_raw+self.reg*self.params['W1']
    grads['b1'] = db1_raw
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class NLayerConvNet(object):

    '''
    With the following structure:
    (conv - relu)xN - 2x2 max pool - affine - relu - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W),
    consisting of N images, each with height H and width W and with C input
    channels.
    '''

    def __init__(self,
                 input_dim = (3,32,32),
                 num_filters = [1,1,1],
                 filter_size = 7,
                 hidden_dim = 100,
                 num_classes = 10,
                 weight_scale = 1e-3,
                 reg = 0.0,
                 dtype = np.float32):

        self.params = {}
        self.len_filters = len(num_filters)
        self.reg = reg
        self.dtype = dtype

        C,H,W = input_dim
        # (conv - relu)xN
        for i in range(self.len_filters):
            if i == 0:
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i],C,filter_size,filter_size)
            else:
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i],num_filters[i-1],filter_size,filter_size)
            self.params['b'+str(i+1)] = np.zeros(num_filters[i])
        # 2x2 max pool - affine - relu
        self.params['W'+str(self.len_filters+1)] = weight_scale*np.random.randn(int(num_filters[self.len_filters-1]*(H/2)*(W/2)),hidden_dim)
        self.params['b'+str(self.len_filters+1)] = np.zeros(hidden_dim)
        # affine - softmax
        self.params['W'+str(self.len_filters+2)] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b'+str(self.len_filters+2)] = np.zeros(num_classes)

        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self,X,y = None):

        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride':1,'pad':int((filter_size-1)/2)}
        pool_param = {'pool_height':2,'pool_width':2,'stride':2}

        # (conv - relu)xN
        current_x = X
        conv_relu_cache = {}
        for i in range(self.len_filters):
            current_x,conv_relu_cache[str(i+1)] = conv_relu_forward(current_x,self.params['W'+str(i+1)],self.params['b'+str(i+1)],conv_param)

        # 2x2 max pool
        current_x,pool_cache = max_pool_forward_fast(current_x,pool_param)
        # affine - relu
        current_x,affine_relu_cache = affine_relu_forward(current_x,self.params['W'+str(self.len_filters+1)],self.params['b'+str(self.len_filters+1)])
        # affine - softmax
        scores,affine_cache = affine_forward(current_x,self.params['W'+str(self.len_filters+2)],self.params['b'+str(self.len_filters+2)])


        if y is None:
            return scores


        loss,grads = 0,{}

        # softmax - affine
        loss,dscores = softmax_loss(scores,y)
        dcurrent_x,dw_raw,db_raw = affine_backward(dscores,affine_cache)
        loss += 0.5*self.reg*np.sum(self.params['W'+str(self.len_filters+2)]*self.params['W'+str(self.len_filters+2)])
        grads['W'+str(self.len_filters+2)] = dw_raw+self.reg*self.params['W'+str(self.len_filters+2)]
        grads['b'+str(self.len_filters+2)] = db_raw

        # relu - affine
        dcurrent_x,dw_raw,db_raw = affine_relu_backward(dcurrent_x,affine_relu_cache)
        loss += 0.5*self.reg*np.sum(self.params['W'+str(self.len_filters+1)]*self.params['W'+str(self.len_filters+1)])
        grads['W'+str(self.len_filters+1)] = dw_raw+self.reg*self.params['W'+str(self.len_filters+1)]
        grads['b'+str(self.len_filters+1)] = db_raw

        # 2x2 max pool backward
        dcurrent_x = max_pool_backward_fast(dcurrent_x,pool_cache)

        # (conv - relu)xN backward
        for i in range(self.len_filters-1,-1,-1):
            dcurrent_x,dw_raw,db_raw = conv_relu_backward(dcurrent_x,conv_relu_cache[str(i+1)])
            loss += 0.5*self.reg*np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
            grads['W'+str(i+1)] = dw_raw+self.reg*self.params['W'+str(i+1)]
            grads['b'+str(i+1)] = db_raw
    
        return loss, grads
from utils import *

def model(n_cnn):
  tf.reset_default_graph()

  ###### Model Parameters ###################
  cnn_size  = [32,32,32,32,32,32,32,32]   ###
  filter_size       = [3,3]               ###
  hidden_layer_size = 512                 ###
  out_dimension     = 2                   ###
  reg_cnn           = 0.1                 ### Regularization weight
  reg_fully         = 1                   ###       "          "
  ###########################################

  with tf.name_scope('Input_Layer'):
    x = tf.placeholder('float', [None, 30*72, 3],'Input_Images')
    x1 = tf.cast(tf.reshape(x,shape=[-1,30,72,3]),tf.float32)
  keep_prob=tf.placeholder(tf.float32)
  #  n_cnn = tf.placeholder(tf.int32)

  output = x1
  reg_cost = tf.constant(0.0)
  for i in range(n_cnn):
    input = output
    output,f = cnn_layer(input,cnn_size[i],[3,3],padding='SAME',filters_summary=True,return_filters=True,function='prelu')
    reg_cost = reg_cost + reg_cnn*tf.reduce_max(tf.square(f)) + reg_cnn*tf.reduce_max(tf.abs(f))
    if(i%2!=0):
      output = maxpool_layer(output)
      mxp_samples = tf.summary.image('MaxPool',tf.slice(put_images_on_grid(output),[0,0,0,0],[-1,-1,-1,1]),1)
      output = dropout(output,keep_prob)

  fc1,f1   = layer(output,hidden_layer_size*2,return_weights=True,function='prelu')
  reg_cost = reg_cost + reg_fully*tf.reduce_mean(tf.square(f1)) + reg_fully*tf.reduce_mean(tf.abs(f1))

  drop1    = dropout(fc1,keep_prob)

  fc2,f2   = layer(drop1,hidden_layer_size,return_weights=True,function='sigmoid')

  output,fo = layer(fc2,out_dimension,name='Output_Layer',function='none',return_weights=True)

  with tf.name_scope('Labels'):
    y = tf.cast(tf.placeholder('float', [None,out_dimension]),tf.float32)

  with tf.name_scope('Train'):
    reg_w     = tf.placeholder(tf.float32)
    not_acc   = tf.reduce_mean(tf.square(output-y))
    cost      = not_acc + reg_w*reg_cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)


  with tf.name_scope('Summaries'):
    all_summaries = tf.summary.merge_all()
    tensorshow(reg_cost,'scalar','Regularization')
    with tf.name_scope('ImageVisualization'):
      in_samples   = tf.summary.image('InputImages', tf.reshape(x,[-1,20,72,3]), 1)
      list = [in_samples] #, mxp2_samples] #, mxp3_samples]
      image_sum    = tf.summary.merge(list)
    loss_sum       = tensorshow(cost,'scalar','TotalLoss')
    trainloss_sum  = tensorshow(not_acc,'scalar','TrainLoss')
    testloss_sum   = tensorshow(not_acc,'scalar','ValidationLoss')

  return x,y,keep_prob,reg_w,not_acc,optimizer,all_summaries,loss_sum,trainloss_sum,testloss_sum,reg_cost,image_sum

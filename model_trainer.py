from utils import *
from model_builder import model
#from model_builder import *
#from broad_model_builder import *

from data_loader_plus import load_data

TRAIN_MODEL_PATH = "Trained Models/"
SHOW_TRAINING_PROGRESS = True # (On tensorboard)
PRINT_PROGRESS = False        # (Print on screen)

#### TRAINING PARAMETERS #####################
from_previous_models = False               ###
epochs        = 1000                       ### Max epochs
batch_sizes   = [128*1,128*2]              ###
dropouts      = [.9,.7,.5]                 ### (Neurons dropout)
reg_weights   = [0.1]                      ###
cnn_numbers   = [6]                        ###
##############################################

### Data loading and processing ##########
pics_each_person = 105*10
data, labels = load_data("Dataset/Columbia_Processed_Color_400_400",show_detailed_process=False,only_eyes=True,horiz_flipping=True,crop=True)
data = np.reshape(data,[56*pics_each_person,30*72,3]).astype(float)
labels = np.reshape(labels,[56*pics_each_person,2]).astype(float)
# Dividing the data in training and validation
train_data = data[5*pics_each_person:]
train_labels = labels[5*pics_each_person:]
valid_data = data[:5*pics_each_person]
valid_labels = labels[:5*pics_each_person]
n_training_data = len(train_data)
##########################################

#print '\nTraining Data Shape:', train_data.shape
#print 'Validation Data shape:', valid_data.shape

for reg_weight in reg_weights:
  for batch_size in batch_sizes:
    for dropout in dropouts:
      for cnn_number in cnn_numbers:
        x,y,keep_prob,reg_w,not_acc,optimizer,all_summaries,loss_sum,trainloss_sum,testloss_sum,reg_cost,image_sum = model(cnn_number)
        saver = tf.train.Saver()
        with tf.Session() as sess:
          model_name  = 'cnn='+str(cnn_number)+'-reg='+str(reg_weight)+'-drop='+str(dropout)+'-batch='+str(batch_size)
          file_writer = tf.summary.FileWriter('Tensorboard/'+model_name+'/')

          file_writer.add_graph(sess.graph)
          sess.run(tf.global_variables_initializer())
          if(from_previous_models):
            try:
              saver.restore(sess=sess,save_path=TRAIN_MODEL_PATH+model_name+'/model.ckpt')
              print 'Found a previous version of trained model!'
            except:
              print 'Starting from zero'
          if(PRINT_PROGRESS):
            print '\nTotal to run:',epochs,'epochs\n'
            print 'Train. loss || Valid. loss'
          for epoch in range(epochs):
            rand_indexes = np.random.random_integers(0, n_training_data-1, n_training_data)
            for batch_n in range(n_training_data/batch_size):
              batch_indexes = rand_indexes[batch_n*batch_size:(batch_n+1)*batch_size]
              batch_data, batch_labels = train_data[batch_indexes], train_labels[batch_indexes]
              _ = sess.run(optimizer, feed_dict=
                {x: batch_data, y: batch_labels, keep_prob: dropout, reg_w: reg_weight})

            if((epoch+1)%100==0):
              print 'Epoch', epoch+1, 'of', epochs, 'completed!'
              saver.save(sess, TRAIN_MODEL_PATH+model_name+'epoch='+str(epoch+1)+'/model.ckpt')
              print("Model "+model_name+"at epoch"+str(epoch+1)+"successfully saved in %s" % TRAIN_MODEL_PATH+model_name+'/model.ckpt')

            if(SHOW_TRAINING_PROGRESS):
              summaries,acc_summary,tot_loss_sum,weights_cost,train_loss = sess.run([all_summaries,trainloss_sum,loss_sum,reg_cost,not_acc], feed_dict=
                {x: train_data[:5*pics_each_person], y: train_labels[:5*pics_each_person], keep_prob: 1, reg_w: reg_weight})
              file_writer.add_summary(tot_loss_sum, epoch)
              file_writer.add_summary(acc_summary, epoch)
              file_writer.add_summary(summaries, epoch)
              summary,valid_loss = sess.run([testloss_sum,not_acc], feed_dict=
                {x: valid_data, y: valid_labels, keep_prob: 1, reg_w: reg_weight})
              file_writer.add_summary(summary, epoch)
              if(PRINT_PROGRESS):
                print "   ", train_loss, "||", valid_loss, "        + reg_cost =", weights_cost
              if(train_loss<=0.01):
                break
	# Visualize the CNN images outputs
          image_samples = sess.run(image_sum, feed_dict=
            {x: train_data, y: train_labels, keep_prob: 1, reg_w: reg_weight})
          file_writer.add_summary(image_samples)

#          saver.save(sess, TRAIN_MODEL_PATH+model_name+'/model.ckpt')
#          print("Model "+model_name+" successfully saved in %s" % TRAIN_MODEL_PATH+model_name+'/model.ckpt')


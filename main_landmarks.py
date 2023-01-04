from train_landmarks import train_and_checkpoints
import os 
import tensorflow as tf
from datasets_landmarks import gen_300WLP_with_landmarks
from model_landmarks import U_net
import pandas as pd 



#datasets

csv_template = os.path.join('..', 'resources', '300W_LP', 'prepro', '{}.csv')
train_path = csv_template.format('train')
valid_path = csv_template.format('valid')
test_path = csv_template.format('test')
full_path = csv_template.format('tout')
print(train_path)


train_dataset = gen_300WLP_with_landmarks(train_path, batchsize =32 )
test_dataset = gen_300WLP_with_landmarks(test_path, batchsize = 32)
valid_dataset = gen_300WLP_with_landmarks(valid_path, batchsize = 32)
full_dataset = gen_300WLP_with_landmarks(full_path, batchsize = 32)


# model

lr = 5e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr,decay_steps=55000,decay_rate=0.96,staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)


model = U_net()
#model = U_net(weights='efficientnetb0_notop.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))


#pre_train = []
#pre_valid = []
#pre_nme = []



li_loss_totale = []
li_loss_valid = []
nme = []

sharded_train = train_dataset.shard(num_shards = 100,index=0)
sharded_valid = valid_dataset.shard(num_shards = 1000,index=0) 
sharded_test = valid_dataset.shard(num_shards = 1000,index=0)

#train_and_checkpoints(1, sharded_train,sharded_valid,test_dataset,pre_train,pre_valid,pre_nme,model,batch_size=32)
#model.load_weights('results_elie/landmarks/ultima5/export_ultima5/weights_final.h5')


epochs = 150
train_and_checkpoints(epochs, full_dataset,sharded_valid,sharded_test, li_loss_totale, li_loss_valid,nme, model,batch_size=32)

# save model complete the path
model.save_weights('___/weights.h5')


# save losses
epo = [k for k in range(1,epochs+1)]


dict = {'epoch':epo, 'loss_train':li_loss_totale}
df1 = pd.DataFrame(dict)
df1.to_csv('____/loss_train.csv')

dict2 = {'epoch':epo, 'loss_valid':li_loss_valid}
df2 = pd.DataFrame(dict2)
df2.to_csv('____/loss_valid.csv')


dict3 = {'epoch':epo, 'nme':nme}   
df3 = pd.DataFrame(dict3)  
df3.to_csv('____/nme.csv')




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np

#add input and target path
#INPUT_PATH = 
#TARGET_PATH = 

momentum = 0.9
nEpochs = 300
batchSize = 128

nFeatures = 39 
nHidden = 512
nClasses = 30 

#DataLoading 
with open('TIMIT_data_prepared_for_CTC.pkl','rb') as f:          #add file name in the blank space
	data= pickle.load(f)

input_list = data['x']
target_list = data['y_indices']
charmap = data['chars']
charmap.append('_')
batchedData, maxTimeSteps = data_lists_to_batches(input_list, target_list, batchSize)
totalN = len(input_list)

graph = tf.Graph()

with tf.variable_scope('CTC'):
	with graph.as_default():
inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))      #defining lstm network
inputXrs = tf.reshape(inputX, [-1, nFeatures])
    		inputList = tf.split(0, maxTimeSteps, inputXrs)
    		targetIxs = tf.placeholder(tf.int64)
    		targetVals = tf.placeholder(tf.int32)
    		targetShape = tf.placeholder(tf.int64)
    		targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    		seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    		weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    		biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    		weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    		biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    		weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    		biasesClasses = tf.Variable(tf.zeros([nClasses]))

       		forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)       #bidirectional lstm
    		backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    		fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
    		fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    		outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    		logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

       		logits3d = tf.pack(logits)                                                      #optimisation
    		loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    		optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    
    		logitsMaxTest = tf.slice(tf.argmax(logits3d,2), [0, 0], [seqLengths[0], 1])              #evaluation
    		predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    		dense_pred = tf.sparse_tensor_to_dense(predictions)
    		errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) 
         
            	local net = nn.Sequential()                                                              #For adding projection layer
            	net:add(nn.SeqLSTM(inputX, forwardH1))
            	net:add(nn.TemporalAdapter(nn.Linear(backwardH1, out1)))   

#Run
with tf.Session(graph=graph) as session:
    print('Initializing')
    saver = tf.train.Saver()
    
    c = tf.train.get_checkpoint_state('/users/TeamASR/models/')
    if c and tf.gfile.Exists(c.model_checkpoint_path):
        print("Reading model parameters from %s" % c.model_checkpoint_path)
        saver.restore(session, c.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())

    
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) 
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths, batchTargetLists = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            [_, l, er, lmt, pred, logit] = session.run([optimizer, loss, errorRate, logitsMaxTest,dense_pred,logits3d], feed_dict=feedDict)
            print(np.unique(lmt))
	    out1 = [charmap[i] for i in lmt ]
	    target1 = [ charmap[i]  for i in batchTargetLists[0] ]
	    o1 = ''.join(out1)
	    t1 = ''.join(target1)

        print('argmax Output: ' + o1)
	    print( 'Target: ' + t1)
        print(pred[0].shape)

            string = [ charmap[i] for i in pred[0] ]
            beam_out = ''.join(string)
            print('Beam Output: '+ beam_out)	    

            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        if(epoch%20 ==19):
		save_path = saver.save(session, '/users/TeamASR/models/mfat@'+str(epoch+1))
		print('model saved in file: '+ save_path)

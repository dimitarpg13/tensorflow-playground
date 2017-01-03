import tensorflow as tf
import os


# initialize variables/model parameters
W = tf.Variable(tf.zeros([5,1]),name="weights")
b = tf.Variable(0., name="bias")

# define the training loop operations

# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b

# new inferred value is the signmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X)) 
  # compute inference model over data X and return the result

def loss(X,Y):
  # compute loss over training data X and expected outputs Y
    #Y_predicted = inference(X)
    #return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X), Y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__filename__) + "/" + file_name])

    reader = tf.TextLineReader(skip_header_lines=1)

    key, value = reader.read(filename_queue)

    # decode_csv will convert a tensorfrom type string (the text line) in a
    # tuple of tensor columns with the specified defaults, which also 
    # sets the data type for each column 

    decoded =  tf.decode_csv(value, record_defaults=record_defaults)
    
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    
    return tf.train.shuffle_batch(decoded, batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0],
                                   [0.0], [""], [0.0], [""], [""]])

    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))

    is_second_class =tf.to+float(tf.equal(pclass, [2]))

    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # finally we pack all the features in a single matrix;
    # we then transpose to have a matrix wih one exampe per row and 
    # one feature per column

    features = tf.transpose(tf.pack([is_first_class, is_second_class, is_third_class,
                            gender, age]))

    survived = tf.reshape(survived, [100, 1])

    return features, survived
 


def train(total_loss):
  # train / adjust model parameters according to computed total loss
    learning_rate = 0.0000001

    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
  # evaluate the resulting trained model
    print (sess.run(inference([[80., 25.]]))) # ~ 303

    print (sess.run(inference([[65., 25.]]))) # ~ 256  

# create a saver
saver =tf.train.Saver()

# launch the graph in session, setup boilerplate
with tf.Session() as sess:
   # model setup
     tf.global_variables_initializer().run()
   
     X, Y = inputs()
     total_loss = loss(X, Y)
     train_op = train(total_loss)

     coord = tf.train.Coordinator()

     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

     # actual training loop
     training_steps = 1000
     initial_step = 0

     # verify if we don't have a checkpoint saved already
     ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
     if ckpt and ckpt.model_checkpoint_path:
       # restores from checkpoint
         saver.restore(sess, ckpt.model_checkpoint_path)
     
         initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])

     
     for step in range(initial_step,training_steps):
         sess.run([train_op])
         # for debugging and learning purposes, see how the loss gets 
         # decremented thru training steps
         if step % 10 == 0:
             print ("loss: ", sess.run([total_loss]))
   
         if step % 1000 == 0:
             saver.save(sess,'my-model', global_step=step)

         # evaluation
         evaluate(sess, X, Y)
      
      
     coord.request_stop()
     coord.join(threads)
     sess.close()


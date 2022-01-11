import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

sess = tf.Session()
new_saver = tf.train.import_meta_graph('models/modelname.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('models'))
all_vars = tf.get_collection('vars')

print("Before the loop..")

for v in all_vars:
    v_ = sess.run(v)
    print("v_:", v_)

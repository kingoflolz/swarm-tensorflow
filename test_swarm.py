import tensorflow as tf
import swarm_session

swarm_sess = swarm_session.SwarmSession(name_prefix="kindiana")
swarm_sess.reconnect_and_create(max_tpus=3)

with tf.compat.v1.Session(swarm_sess.get_target()) as sess:
    # create explicit session as its easier to pass around different threads
    print(sess.list_devices())

swarm_sess.delete_swarm()
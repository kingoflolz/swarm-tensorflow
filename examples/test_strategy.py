import threading
import time

import swarm_session
import tensorflow as tf
import numpy as np

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver("kindiana_e52bdc0072f3684c288f5280a7555a5f")
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)

swarm_sess = swarm_session.SwarmSession(name_prefix="kindiana")
swarm_sess.reconnect_and_create(max_tpus=2)
strategy = swarm_sess.strategies[swarm_sess.tpus[0]]

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# with tf.compat.v1.Session(swarm_sess.get_target()) as sess:
def dataset_fn():
    x = np.random.random((2, 5)).astype(np.float32)
    y = np.random.randint(2, size=(2, 1))
    return x, y

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(5,)),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

@tf.function(experimental_compile=False)
def step_fn():
    with tf.device(f"/job:{swarm_sess.tpus[0]}/device:TPU:*"):
        features = tf.random.normal((2, 5))
        labels = tf.random.uniform((2, 1), maxval=2, dtype=tf.dtypes.int32)
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

@tf.function(experimental_compile=False)
def step_fn2():
    with tf.device(f"/job:{swarm_sess.tpus[0]}/device:TPU:*"):
        strategy.run(step_fn)

with tf.device(f"/job:{swarm_sess.tpus[0]}"):
    for i in range(10):
        step_fn2()
        print(f"tpu 0 train step {i}")
        time.sleep(0.1)

swarm_sess.delete_swarm()
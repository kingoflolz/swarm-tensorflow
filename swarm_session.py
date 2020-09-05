import os
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"

import secrets
from cloud_tpu_client import Client
import tpunicorn
from multiprocessing.pool import ThreadPool
import logging

import psutil
from mock import patch
from tpu_system_metadata import _query_tpu_system_metadata as _query_tpu_system_metadata_patched

patcher = patch('tensorflow.python.tpu.tpu_system_metadata._query_tpu_system_metadata', _query_tpu_system_metadata_patched)
patcher.start()

import tensorflow as tf
tf.debugging.set_log_device_placement(True)
tf.get_logger().setLevel(logging.INFO)

class SwarmSession():
    def __init__(self, name_prefix="swarm", tpu_version=tf.__version__, api_threads=8):
        self.tpus = []
        self.tpu_resolvers = {}
        self.strategies = {}
        self.topologies = {}
        self.name_prefix = name_prefix
        self.total_tpu_count = 0
        self.tpu_version = tpu_version
        self.pool = ThreadPool(api_threads)

    # reconnect to TPUs which match the name prefix
    def reconnect_tpus(self, max_tpus=None):
        connect_tpus = [tpu["name"].split("/")[-1] for tpu in tpunicorn.get_tpus() if tpu["name"].split("/")[-1].startswith(self.name_prefix)]

        # self.pool.map(self.connect_tpu, connect_tpus)

        if max_tpus is not None:
            connect_tpus = connect_tpus[:max_tpus]

        for tpu in connect_tpus:
            self.connect_tpu(tpu)

    def connect_tpu(self, tpu_name):
        self.tpus.append(tpu_name)

        c = Client(tpu_name)
        c.wait_for_healthy(interval=5)
        c.configure_tpu_version(self.tpu_version, restart_type='ifNeeded')
        c.wait_for_healthy(interval=5)

        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name, job_name=tpu_name)
        self.tpu_resolvers[tpu_name] = tpu_resolver

        if len(self.tpu_resolvers) == 1:
            tf.config.experimental_connect_to_cluster(tpu_resolver)
        else:
            # TODO: might want to check health of nodes in self.tpus before redefining the cluster
            tf.config.experimental_connect_to_cluster(tf.distribute.cluster_resolver.UnionResolver(*self.tpu_resolvers.values()))
        topology = tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        self.strategies[tpu_name] = tf.distribute.TPUStrategy(tpu_resolver)

        self.topologies[tpu_name] = topology
        self.total_tpu_count = len(self.tpus)

    # TODO replace this garbage with tpunicorn api
    def create_tpu(self, count, zone="europe-west4-a", accelerator_type="v3-8", project="youdreamof-1543654322305"):
        new_tpus = []
        commands = []
        for _ in range(count):
            name = f"{self.name_prefix}_{secrets.token_hex(16)}"
            new_tpus.append(name)
            self.total_tpu_count += 1

            commands.append(f"gcloud compute tpus create {name} --zone {zone} --project {project} --version 1.15.2 --accelerator-type {accelerator_type} --quiet")

        print(f"creating TPUs {new_tpus}")
        self.pool.map(os.system, commands)
        print(f"connecting to TPUs {new_tpus}")
        # self.pool.map(self.connect_tpu, new_tpus)
        for tpu in new_tpus:
            self.connect_tpu(tpu)
        print(f"connected to TPUs {new_tpus}")

    def reconnect_and_create(self, max_tpus, zone="europe-west4-a", accelerator_type="v3-8", project="youdreamof-1543654322305"):
        self.reconnect_tpus(max_tpus)
        print(f"reconnected to TPUs {self.tpus}")
        print(f"creating {max_tpus - self.total_tpu_count} more TPUs")
        self.create_tpu(max_tpus - self.total_tpu_count, zone, accelerator_type, project)

    def delete_swarm(self, zone="europe-west4-a", project="youdreamof-1543654322305"):
        commands = []
        for name in self.tpus:
            commands.append(f"gcloud compute tpus delete {name} --zone {zone} --project {project} --quiet")
        self.pool.map(os.system, commands)

    # TODO: this is _really_ ugly, but I can't find the API to do it any other way...
    def get_target(self):
        p = psutil.Process(os.getpid())
        listen_ports = [i for i in p.connections() if i.status == "LISTEN"]
        assert len(listen_ports) == 1

        return f"grpc://localhost:{listen_ports[0].laddr.port}"

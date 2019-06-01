"""
Optimizes a saved judge moded.

Uses steps decribed in:
https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf
"""

import argparse
from datetime import datetime
import os
import sys
import tensorflow as tf
from tensorflow import data
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph
from judge import MNISTJudge, FashionJudge

TRANSFORMS = [
    "remove_nodes(op=Identity)",
    "fold_constants(ignore_errors=true)",
    "merge_duplicate_nodes",
    "strip_unused_nodes",
    "fold_batch_norms",
]


def get_graph_def_from_saved_model(saved_model_dir):
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session, tags=[tag_constants.SERVING], export_dir=saved_model_dir
        )
    return meta_graph_def.graph_def


def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ""
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print("graph freezed!")


def optimize_graph(model_dir, graph_filename, transforms, output_node):
    input_names = []
    output_names = [output_node]
    if graph_filename is None:
        graph_def = get_graph_def_from_saved_model(model_dir)
    else:
        graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
        graph_def, input_names, output_names, transforms
    )
    tf.train.write_graph(
        optimized_graph_def, logdir=model_dir, as_text=False, name="optimized_model.pb"
    )
    print("Graph optimized!")


def convert_graph_def_to_saved_model(
    export_dir, graph_filepath, output_key, output_node_name
):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name="")
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                node.name: session.graph.get_tensor_by_name("{}:0".format(node.name))
                for node in graph_def.node
                if node.op == "Placeholder"
            },
            outputs={output_key: session.graph.get_tensor_by_name(output_node_name)},
        )
        print("Optimized graph converted to SavedModel!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_in", type=str, help="Path to read the model from")
    parser.add_argument(
        "model_out", type=str, help="Path to write the optimized model to"
    )
    args = parser.parse_args()
    saved_model_in = args.model_in

    # freeze model and describe it
    freeze_model(saved_model_in, "softmax_tensor", "frozen_model.pb")
    frozen_filepath = os.path.join(saved_model_in, "frozen_model.pb")

    # optimize model and describe it
    optimize_graph(saved_model_in, "frozen_model.pb", TRANSFORMS, "softmax_tensor")
    optimized_filepath = os.path.join(saved_model_in, "optimized_model.pb")

    # convert to saved model and output metagraph again
    convert_graph_def_to_saved_model(
        args.model_out, optimized_filepath, "softmax_tensor", "softmax_tensor:0"
    )

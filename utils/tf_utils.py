# Author: Zhengying Liu
# Creation Date: 20 Jan 2020
# Description: Functions for visualizing TF grah in Jupyter Notebook. borrowed
#   from:
#     https://github.com/tensorflow/examples/blob/master/community/en/r1/deepdream.ipynb

from IPython.display import clear_output, Image, display, HTML
import numpy as np
import tensorflow as tf

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped {} bytes>".format(size).encode()
    return strip_def


def to_html(graph_def, max_const_size=32, width=1000, height=620):
    """Convert TensorBoard summary to HTML."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:{}px;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(width, height, code.replace('"', '&quot;'))
    return iframe


def show_graph(graph_def, max_const_size=32, width=1000, height=620):
    """Visualize TensorFlow graph."""
    iframe = to_html(graph_def, max_const_size=max_const_size,
                     width=width, height=height)
    display(HTML(iframe))


def show_default_graph(width=1000, height=620):
    graph_def = tf.get_default_graph().as_graph_def()
    show_graph(graph_def, width=width, height=height)


def save_graph_to_html(filename, width=1200, height=620):
    """Save HTML file corresponding to current TensorFlow file."""
    iframe = to_html(tf.get_default_graph().as_graph_def(),
                     width=width, height=height)
    if not filename.endswith('.html'):
        filename += '.html'
    with open(filename, 'w') as f:
        f.write(iframe)


# End block for visualization
class VisulizeGraphHook(tf.train.SessionRunHook):

  def __init__(self, html_path):
    self.html_path = html_path

  def begin(self):
    save_graph_to_html(self.html_path)

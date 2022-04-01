doc(title="Data Loading",
    underline_char="=",
    entries=[
        doc_entry("external_input.ipynb",
                  op_reference("fn.external_source", "Intro tutorial for external source")),
        doc_entry(
            "parallel_external_source.ipynb",
            op_reference("fn.external_source", "How to use parallel mode for external source")),
        doc_entry(
            "parallel_external_source_fork.ipynb",
            op_reference("fn.external_source",
                         "How to use parallel mode for external source in fork mode")),
        doc_entry("dataloading_lmdb.ipynb", [
            op_reference("fn.readers.caffe", "Loading Caffe and Caffe 2 LMBD data"),
            op_reference("fn.readers.caffe2", "Loading Caffe and Caffe 2 LMBD data")
        ]),
        doc_entry("dataloading_recordio.ipynb",
                  op_reference("fn.readers.mxnet", "Loading MXNet recordIO data")),
        doc_entry("dataloading_tfrecord.ipynb",
                  op_reference("fn.readers.tfrecord", "Loading TensorFlow TFRecord data")),
        doc_entry("dataloading_webdataset.ipynb",
                  op_reference("fn.readers.webdataset",
                               "Loading data stored in Webdataset format")),
        doc_entry("coco_reader.ipynb",
                  op_reference("fn.readers.coco", "Loading COCO dataset or its subsets")),
        doc_entry(
            "numpy_reader.ipynb",
            op_reference(
                "fn.readers.numpy",
                "Loading Numpy array files (*.npy) to CPU or directly to GPU with GPUDirect Storage"
            )),
    ])

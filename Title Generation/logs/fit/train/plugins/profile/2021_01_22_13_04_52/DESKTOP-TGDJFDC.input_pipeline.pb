	�q���@�q���@!�q���@	�d��:��?�d��:��?!�d��:��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�q���@_)�Ǻ�?AA��ǘ�@Y�I+��?*	gffffFQ@2U
Iterator::Model::ParallelMapV2Dio��ɔ?!�HԱ`=@)Dio��ɔ?1�HԱ`=@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���H�?!�[S�7@)���H�?1�[S�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%u��?!���/F5@)�HP��?1���ӧ1@:Preprocessing2F
Iterator::Model�X�� �?!���3E@);�O��n�?1���*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ�|a2�?!�.v���=@)a2U0*�s?1�L�X+�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8��d�`�?!|��L@)-C��6j?1 3���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�o�&��@){�G�zd?1�o�&��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw-!�l�?!y��pu�?@)a2U0*�S?1�L�X+��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�d��:��?Im��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	_)�Ǻ�?_)�Ǻ�?!_)�Ǻ�?      ��!       "      ��!       *      ��!       2	A��ǘ�@A��ǘ�@!A��ǘ�@:      ��!       B      ��!       J	�I+��?�I+��?!�I+��?R      ��!       Z	�I+��?�I+��?!�I+��?b      ��!       JCPU_ONLYY�d��:��?b qm��X@
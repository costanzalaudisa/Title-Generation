	}гY�y@}гY�y@!}гY�y@	Hu�ݓ�?Hu�ݓ�?!Hu�ݓ�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$}гY�y@a2U0*��?AW[���@Y���~�:�?*	    �D@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!�h���C@@)46<�R�?1�h���C:@:Preprocessing2F
Iterator::Model������?!�b��7�D@)"��u���?1���k�4@:Preprocessing2U
Iterator::Model::ParallelMapV2�� �rh�?!�M�_{4@)�� �rh�?1�M�_{4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatey�&1�|?!�ˊ��0@)"��u��q?1���k�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�HP��?!N�_{�eM@)��_vOf?1��C.+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_vOf?!��C.+@)��_vOf?1��C.+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!y���k@)��_�Le?1y���k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	�^)ˀ?!rY1P�3@)a2U0*�S?1o4u~�!@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Hu�ݓ�?I��D��X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	a2U0*��?a2U0*��?!a2U0*��?      ��!       "      ��!       *      ��!       2	W[���@W[���@!W[���@:      ��!       B      ��!       J	���~�:�?���~�:�?!���~�:�?R      ��!       Z	���~�:�?���~�:�?!���~�:�?b      ��!       JCPU_ONLYYHu�ݓ�?b q��D��X@
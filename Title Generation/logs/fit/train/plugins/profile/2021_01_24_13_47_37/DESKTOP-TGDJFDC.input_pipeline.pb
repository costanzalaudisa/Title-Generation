	��v��@��v��@!��v��@	9��W�s�?9��W�s�?!9��W�s�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��v��@��o_�?A.�!��u	@Y��镲�?*����̌H@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL7�A`�?!��-�l�@@)lxz�,C�?1A��<@:Preprocessing2F
Iterator::Model"��u���?! ��A@)"��u���?1 ��1@:Preprocessing2U
Iterator::Model::ParallelMapV2"��u���?! ��1@)"��u���?1 ��1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�|a2U�?!��px>P@)��_vOv?1���%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��y�):�?!�:�kS 2@)�g��s�u?1�7$��%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!|�f�S@)��H�}m?1|�f�S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!���@)��_vOf?1���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{�G�z�?!�G�j�]4@)/n��R?1Gh��/�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9:��W�s�?I�P{�X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��o_�?��o_�?!��o_�?      ��!       "      ��!       *      ��!       2	.�!��u	@.�!��u	@!.�!��u	@:      ��!       B      ��!       J	��镲�?��镲�?!��镲�?R      ��!       Z	��镲�?��镲�?!��镲�?b      ��!       JCPU_ONLYY:��W�s�?b q�P{�X@
	�� �r�@�� �r�@!�� �r�@	y���bm�?y���bm�?!y���bm�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�� �r�@=�U����?A��u��@Y��\m���?*	33333�H@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�q����?!��x�4C?@)9��v���?1_"��V:@:Preprocessing2F
Iterator::ModelˡE����?!�=X\��D@)�g��s��?1Щ�~>5@:Preprocessing2U
Iterator::Model::ParallelMapV2��ׁsF�?!Q��9��3@)��ׁsF�?1Q��9��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��y�):�?!�	����1@)Ǻ���v?1��K�q&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!p§�{uM@)ŏ1w-!o?19�T��u@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u�k?!���s@)F%u�k?1���s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!���x�@)��_�Le?1���x�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{�G�z�?!�ˁ�B
4@)/n��R?1d��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9y���bm�?I�duJ�X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	=�U����?=�U����?!=�U����?      ��!       "      ��!       *      ��!       2	��u��@��u��@!��u��@:      ��!       B      ��!       J	��\m���?��\m���?!��\m���?R      ��!       Z	��\m���?��\m���?!��\m���?b      ��!       JCPU_ONLYYy���bm�?b q�duJ�X@
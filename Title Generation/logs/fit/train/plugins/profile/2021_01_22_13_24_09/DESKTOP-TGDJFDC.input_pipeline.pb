	R'���a@R'���a@!R'���a@	LcON~�?LcON~�?!LcON~�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$R'���a@��ZӼ��?A����o@Y��|гY�?*	33333�L@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�q����?!5؝�h�:@)-C��6�?1?;��i6@:Preprocessing2F
Iterator::Model�+e�X�?!�l.j�C@)Zd;�O��?1;i���3@:Preprocessing2U
Iterator::Model::ParallelMapV2M�St$�?!Np�I3�3@)M�St$�?1Np�I3�3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�]K�=�?!�)��{�6@)a2U0*��?1p��@��0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�-����?!<���ON@)a2U0*�s?1p��@�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!���.��@)���_vOn?1���.��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!�s~v�W@)Ǻ���f?1�s~v�W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!��[{c9@)Ǻ���V?1�s~v�W@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9LcON~�?I�s���X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ZӼ��?��ZӼ��?!��ZӼ��?      ��!       "      ��!       *      ��!       2	����o@����o@!����o@:      ��!       B      ��!       J	��|гY�?��|гY�?!��|гY�?R      ��!       Z	��|гY�?��|гY�?!��|гY�?b      ��!       JCPU_ONLYYLcON~�?b q�s���X@
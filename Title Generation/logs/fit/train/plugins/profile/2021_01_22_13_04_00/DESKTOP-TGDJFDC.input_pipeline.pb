	���x�@���x�@!���x�@	�8�ί�?�8�ί�?!�8�ί�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���x�@�W�2ı�?A�[ Aq@YǺ���?*	effff�H@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃎?!
&sp�=@)��@��ǈ?1ڌ�L8@:Preprocessing2F
Iterator::Model�5�;Nё?!D.+JxA@);�O��n�?1��ˊ�2@:Preprocessing2U
Iterator::Model::ParallelMapV2�J�4�?!�ˊ��0@)�J�4�?1�ˊ��0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�HP��?!���f|8@)vq�-�?1#NT��/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!^K/�D!@)"��u��q?1^K/�D!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ǘ���?!�h���CP@)ŏ1w-!o?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!�dn}@)Ǻ���f?1�dn}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�߾�?!>�����;@)�~j�t�X?1�k��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�8�ί�?I��@�X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�W�2ı�?�W�2ı�?!�W�2ı�?      ��!       "      ��!       *      ��!       2	�[ Aq@�[ Aq@!�[ Aq@:      ��!       B      ��!       J	Ǻ���?Ǻ���?!Ǻ���?R      ��!       Z	Ǻ���?Ǻ���?!Ǻ���?b      ��!       JCPU_ONLYY�8�ί�?b q��@�X@
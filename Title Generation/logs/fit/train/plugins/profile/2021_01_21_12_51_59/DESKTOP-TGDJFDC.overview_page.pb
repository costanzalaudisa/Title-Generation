�	j�q��d
@j�q��d
@!j�q��d
@	�m��?�m��?!�m��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j�q��d
@/n���?AH�}8�@Y�z6�>�?*	����̌L@2U
Iterator::Model::ParallelMapV2/�$��?!��؇cB@)/�$��?1��؇cB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!��}E�J7@)��_vO�?1����2@:Preprocessing2F
Iterator::Modelu���?!i���.J@)��y�):�?1�7L�b,/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen��t?![O��)!@)n��t?1[O��)!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ����?!:���I0@)/n��r?1�~_ѷ�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!��~_�G@)-C��6j?1�-.;�j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�<�\�@){�G�zd?1�<�\�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!:�x�2�2@)Ǻ���V?1hȳ]�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�m��?I�IVp��X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/n���?/n���?!/n���?      ��!       "      ��!       *      ��!       2	H�}8�@H�}8�@!H�}8�@:      ��!       B      ��!       J	�z6�>�?�z6�>�?!�z6�>�?R      ��!       Z	�z6�>�?�z6�>�?!�z6�>�?b      ��!       JCPU_ONLYY�m��?b q�IVp��X@Y      Y@q&c�W�S@"�	
both�Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�78.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 
�	2U0*�@2U0*�@!2U0*�@	b��%�"@b��%�"@!b��%�"@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$2U0*�@������?Au�V�@Yo��ʡ�?*	������y@2F
Iterator::ModelF��_��?!jiiii)W@)^�I+�?1��-�W�U@:Preprocessing2U
Iterator::Model::ParallelMapV2��ͪ�Ֆ?!qݻc�@)��ͪ�Ֆ?1qݻc�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvq�-�?!�����x@)-C��6�?1�1��į@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU���N@�?!�lh}!@)�����w?12݀NJ_�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Zd;�?!jiiiii@)�J�4q?1�P]Y3�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!��X���?)��H�}m?1��X���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!�N�t�$�?)�~j�t�h?1�N�t�$�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�$��?!��e/@@)/n��R?1I�aD���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9b��%�"@I�3�B��V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "      ��!       *      ��!       2	u�V�@u�V�@!u�V�@:      ��!       B      ��!       J	o��ʡ�?o��ʡ�?!o��ʡ�?R      ��!       Z	o��ʡ�?o��ʡ�?!o��ʡ�?b      ��!       JCPU_ONLYYb��%�"@b q�3�B��V@Y      Y@q)�}e�EV@"�

both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nohigh"t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb�89.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 
�	�J�4@�J�4@!�J�4@	�/h3�?�/h3�?!�/h3�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�J�4@mV}��b�?A��@��G	@YDio��ɤ?*	      I@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃎?!������=@)tF��_�?1������7@:Preprocessing2F
Iterator::ModelA��ǘ��?!333333F@)M�St$�?1������6@:Preprocessing2U
Iterator::Model::ParallelMapV246<�R�?!������5@)46<�R�?1������5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ǘ���?!3333330@){�G�zt?1      $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Pk�w�?!������K@)F%u�k?1gfffff@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea��+ei?!������@)a��+ei?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!      @)�~j�t�h?1      @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*��?!3333333@)�~j�t�X?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�/h3�?IB�_�3�X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	mV}��b�?mV}��b�?!mV}��b�?      ��!       "      ��!       *      ��!       2	��@��G	@��@��G	@!��@��G	@:      ��!       B      ��!       J	Dio��ɤ?Dio��ɤ?!Dio��ɤ?R      ��!       Z	Dio��ɤ?Dio��ɤ?!Dio��ɤ?b      ��!       JCPU_ONLYY�/h3�?b qB�_�3�X@Y      Y@q{�>�u�T@"�	
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�82.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 
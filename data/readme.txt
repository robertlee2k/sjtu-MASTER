Thank you for your attention to our work.
We are not planning to publish the data infrastructure for explained reasons. If you need to process raw data,
we highly recommend reusing the Qlib implementations. Here is the configuration:

infer_processors:
- class: RobustZScoreNorm
kwargs:
fields_group: feature
clip_outlier: true
- class: Fillna
kwargs:
fields_group: feature
learn_processors:
- class: DropnaLabel
- class: DropExtremeLabel
kwargs:
percentile: 0.975
- class: CSZscoreNorm
kwargs:
fields_group: label

Please note that, except for DropExtremeLabel, the above configuration is used for many models
in qlib/examples/benchmarks and we do use the Qlib implementations in producing the published
dl_train, dl_valid, and dl_test. The DropExtremeLabel is implemented in our commercial codebase,
which should be easy to implement in Qlib as well, since it obeys a simple rule to drop 2.5% of
the highest/lowest labels.
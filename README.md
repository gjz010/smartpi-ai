SmartPi AI Training Script
========

Training
--------


```
# with requirements-train.txt
# You have to augment learning rate in model.py.
python3 train.py
```

Exporting
--------

```
# with requirements-export.txt
python3 export_keras_model.py
# You may want to use --reverse_input_channels if you want to input by RGB instead of BGR.
mo_tf.py --input_model inference_graph.pb --input_shape [10,192,192,3] --mean_values=[127.5,127.5,127.5] --scale_values=[127.5,127.5,127.5]
```

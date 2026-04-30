# A visual analytics framework for quantifying the exploratory behavior of multi-columnar neurons in _Drosophila_
Accompanying repository for our publication of the Drosophila multi-columnar cell analysis tool. Please read the [publication](XYZ) to familiarize yourself with the tool.

## Workflow use
1. Set the input/output path and desired processing parameters in the configuration file.
2. Sequentially execute the _.py_ scripts found in the _src_ folder.

   _E.g._:
   `
   python 00_get_raw_images.py my_config_file.json
   `

   If no configuration filename is provided, the script expects a configuration file named _default_config.json_.
4. Inspect, correct or perform required manual work on the intermediate results in an imaging software of your choice.
5. Visualize and interactively explore the results using the _Column Explorer_.

   Run the _Column Explorer_ using:
   `
   COMMAND HERE
   `

## Example data
Final results of processing for a single dataset are provided in _example_ folder. They can be explored using the _Column Explorer_ without any processing.

## Cite
If using our approach, please cite our [publication](XYZ):
```
```

```bibtex

```

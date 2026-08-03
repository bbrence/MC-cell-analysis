# An end-to-end framework for quantifying the exploratory behavior of multicolumnar neurons in _Drosophila_
Accompanying repository for our publication of the Drosophila multicolumnar cell analysis tool. Please read the [publication](XYZ) to familiarize yourself with the tool.

## Installation
Clone the repository, create a virtual environment, and install the Python dependencies:

```bash
git clone https://github.com/bbrence/MC-cell-analysis.git
cd MC-cell-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Workflow use
1. Set the input/output path and desired processing parameters in the configuration file.
2. Sequentially execute the _.py_ scripts found in the _src_ folder.

   For example, from the repository root:

   ```bash
   python src/00_get_raw_images.py /path/to/my_config_file.json
   ```

   If no configuration filename is provided, the script looks for `default_config.json` in the current working directory. To use the provided default configuration, first edit `src/default_config.json`, then run the scripts from `src`, for example:

   ```bash
   cd src
   python 00_get_raw_images.py
   ```
3. Inspect, correct, or perform required manual work on the intermediate results in an imaging software of your choice.
4. Visualize and interactively explore the results using the _Column Explorer_.

   Run the _Column Explorer_ using:

   ```bash
   panel serve src/column_explorer.py --show
   ```

   In the application, enter the workflow output directory in **Input base folder** and select **Load input folder**.

## Example data
Limited final results of processing for a single dataset are provided in _example-output_ folder (some folders are populated with dummy data to preserve the folder structure created by the workflow). They can be explored using the _Column Explorer_ without any processing.

## Cite
If using our approach, please cite our [publication](XYZ):
```
```

```bibtex

```

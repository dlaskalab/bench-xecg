# bench-xecg

Official repository of "BenchECG and xECG: a benchmark and baseline for ECG foundation models"

📄 Paper: [npj Digital Medicine](https://www.nature.com/articles/s41746-026-03196-y)

🤗 Model: [HuggingFace](https://huggingface.co/riccardolunelli/xECG_base_model_v1)

## How to use xECG

We share the weights for our xECG on [HuggingFace](https://huggingface.co/riccardolunelli/xECG_base_model_v1) along with some convinient classes for downstream tasks. 

We strongly suggest to use the xECG class from there, as we simplified the code: in this repository it might be less intuitive how to use xECG.

## How to evaluate on BenchECG

### Adapt your model to our pipeline

Our code comes with a convenient `models.BaseModel` class. This class has some methods that are used by the trainer to setup the optmizer: defining different learning rates for different part of the network (e.g. layerwise decay).

`models.BaseModel` assumes models extending it are composed by a pre-trained core and a head appended to it for the downstream task.

To easily add a new model to our pipeline follow these steps:
- Your model should inherit from `models.BaseModel`
- Add a variable on the configuration (e.g. `use_your_model`) and set it to true
- In `utils.utils` modify `get_base_model` to properly load your model. That function comes with the parameters `feature_classification`. Look at `models.classification.py` for an example.
- Again in `utils.utils` add to `parse_config` the variable you chose for your model and define there `sampling_freq`, `patch_size`, and some eventual default configuration specific to your model and that will not change with different tasks (e.g. preprocessing hyperparams)

We suggest you to be authenticated with `wandb` to see the logs.

If you need to use layerwise decay: 
- Implement the functions `get_layers`, this allow to use layerwise decay
- Implement `additional_params`, again for layerwise decay define all the parameters that are not in the layer list, they will have the smallest learning rate.

### Find optimal hyperparameters on downstream tasks

After the model is adapted to the pipeline, for each task create two different configuration files, one for linear probing and one for fine-tuning. 
In the `config` folder there are the configuration files used in our work to use as a reference.

Details of each task are in the following documentations:
- [PTB-XL](docs/ptbxl.md)
- [CPSC2018](docs/cpsc2018.md)
- [MIT-BIH (classification)](docs/mit_bih.md)
- [MIT-BIH (R-peak detection)](docs/r_peak.md)
- [Sleep Apnea](docs/sleep_apnea.md)
- [PPG AF](docs/ppg_af.md)
- [Exercise (R-peak detection)](docs/exercise.md)
- [Age](docs/age.md)
- [Blood test](docs/blood_test.md)
- [Survival](docs/survival.md)

### Launch all the experiments

Once the individual configuration files are created, prepare a task list file (e.g., `configs/my_model_tasks.yaml`) that references all the experiments you want to run for both linear probing and fine-tuning. You can use `configs/task_config_list_lp.yaml` as a template.

To ensure consistent logging, use the same wandb_group value for all tasks within a specific experiment type:
- linear probing: `wandb_group: 'your_model_lp'`
- fine-tuning: `wandb_group: 'your_model_ft'`

To launch the experiments, run:
```shell
uv run train_all.py --config_file configs/<your-config-list>.yaml --num_runs 5
```

### Get the bench ecg score

After all the experiemnts ended successfully run the following to get the final *BenchECG Score*:

```shell
uv run benchscore.py logs_lightning <your-experiment-name>
```

## Call for feedbacks

We really appreciate any issue / suggestion / feedback to improve our repository!

## Disclaimer

The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use. The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. 
In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## Citation

If you use our BenchECG code, xECG model or just find our code helpful, please cite:
```
@article{lunelliBenchECGXECGBenchmark2026,
  title = {{{BenchECG}} and {{xECG}}: A Benchmark and Baseline for {{ECG}} Foundation Models},
  shorttitle = {{{BenchECG}} and {{xECG}}},
  author = {Lunelli, Riccardo and Nicolson, Angus and Pröll, Samuel Martin and Reinstadler, Sebastian Johannes and Bauer, Axel and Dlaska, Clemens},
  date = {2026-09-14},
  journaltitle = {npj Digital Medicine},
  shortjournal = {Npj Digit. Med.},
  publisher = {Nature Publishing Group},
  issn = {2398-6352},
  doi = {10.1038/s41746-026-03196-y},
  url = {https://www.nature.com/articles/s41746-026-03196-y},
  urldate = {2026-09-17},
  langid = {english},
  keywords = {Cardiology,Computational biology and bioinformatics,Health care,Mathematics and computing},
}
```

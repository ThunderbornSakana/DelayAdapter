# DelayAdapter
Addressing Delayed Feedback in Conversion Rate Prediction: A Domain Adaptation Approach.

## Requirements
- Python 3.11.7
- PyTorch 2.1.2

## Data Downloading and Preprocessing
First, make the following directories:
- `./data/criteo`
- `./data/taobao`
- `./data/tencent`
- `./checkpoints`

Then, download Criteo from [Criteo](https://labs.criteo.com/2013/12/conversion-logs-dataset/) and put raw data in `./data/criteo`; Similarly, download Taobao and Tencent from [Taobao](https://tianchi.aliyun.com/dataset/649) and [Tencent](https://link.juejin.cn/?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1JnP4Vvr-6HKYlG5bNlfvMQ) (code: ebd2), respectively.

Decide which train-test split to generate and change the field `use` to `true` in the corresponding dictionary in `./src/preprocessor/configs.json`. Then, in `./src/preprocessor`, run the following command:
```bash
python run.py
```

## Model Training and Evaluation
In `./src/configs.json`, get the `tag` ID of the hyperparameter combination that you want to experiment with. You can run new experiments by creating new tags. For example, if we want to run the experiment with `tag` ID as `0`, the following command can be run in `./src`:
```bash
python main.py --device_idx 0 --tag 0
```
The evaluation results at each epoch are saved under `./checkpoints`.

## Choosing Optimal Hyperparameters
Tunable hyperparameters include the choice of base models and UDA algorithms, their respective hyperparameters, and $H$. To simplify experimentation, hyperparameters specific to base models are only optimized once for Naive. UDA-specific hyperparameters are initially tuned independently, followed by joint optimization of UDA and base model choices along with other parameters (determined through grid search). We present the tuning ranges for all hyperparameters across three datasets, assuming that selections for base models and UDA algorithms have already been made for simplicity.

We first show the ranges for the Criteo dataset. We chose AutoInt as the base model and MDD as the UDA algorithm for Criteo.

| Hyperparameters                 | Tuning Ranges                 |
| --------                        | --------                      |
| Emebdding dimension             |  128, 256                     |
| Hidden dimension                |  64, 128, 256                 |
| Number of attention heads       |  4, 6, 8                      |
| Number of self-attention layers |  2, 3, 4                      |
| Dropout                         |  0.1, 0.3, 0.5                |
| lambda (MDD)                    |  0.01, 0.05, 0.1              |
| $H$                             |  2, 6, 10, 14, 18, 22, 26, 30 |

Then we show the ranges for the Taobao dataset. We chose MLP as the base model and DANN as the UDA algorithm for Taobao.

| Hyperparameters                 | Tuning Ranges                 |
| --------                        | --------                      |
| Emebdding dimension             |  128, 256                     |
| Hidden dimension                |  64, 128, 256                 |
| Number of layers (MLP)          |  2, 3, 4                      |
| Dropout                         |  0.1, 0.3, 0.5                |
| lambda (DANN)                   |  0.01, 0.05, 0.1              |
| $H$                             |  0.5, 1, 1.5, 2, 2.5          |

Finally we show the ranges for the Tencent dataset. We chose FiBiNET as the base model and DANN as the UDA algorithm for Tencent.

| Hyperparameters                 | Tuning Ranges                 |
| --------                        | --------                      |
| Emebdding dimension             |  128, 256                     |
| Hidden dimension                |  64, 128, 256                 |
| Number of layers (FiBiNET)      |  2, 3, 4                      |
| Dropout                         |  0.1, 0.3, 0.5                |
| lambda (DANN)                   |  0.0005, 0.001, 0.005, 0.01   |
| $H$                             |  0.5, 1, 1.5, 2, 2.5          |

The best combinations of hyperparameters on all three datasets can be found in the **configs.json file in our open-sourced code**.

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

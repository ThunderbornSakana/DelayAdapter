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

Then, download Criteo from [Criteo](https://labs.criteo.com/2013/12/conversion-logs-dataset/) and put raw data in `./data/criteo`; Similarly, download Taobao and Tencent from [Taobao](https://tianchi.aliyun.com/dataset/649) and [Tencent](https://link.juejin.cn/?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1JnP4Vvr-6HKYlG5bNlfvMQ)(code: ebd2), respectively.

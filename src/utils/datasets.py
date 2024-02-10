import torch.utils.data as data

NUM_SEC_PER_MINUTE = 60
NUM_SEC_PER_HOUR = NUM_SEC_PER_MINUTE * 60

class CVR_Dataset(data.Dataset):
    def __init__(self, cat, label, test_start_ts, convert_ts, num=None):
        self.num = num
        self.cat = cat
        self.oracle = label.copy()
        self.y = label.copy()
        self.y[convert_ts >= (test_start_ts - NUM_SEC_PER_HOUR)] = 0.0

    def __len__(self):
        return len(self.oracle)

    def __getitem__(self, idx):
        if self.num is not None:
            return self.num[idx], self.cat[idx], self.oracle[idx], self.y[idx]
        else:
            return False, self.cat[idx], self.oracle[idx], self.y[idx]

from master import MASTERModel
import pickle
import os
import pandas as pd

class DataArgs():
    is_windows = os.name == 'nt'
    default_qlib_data_path = (
        '../InvariantStock/data/qlib_data/cn_data' if is_windows else
        '../.qlib/qlib_data/cn_data'
    )
    data_pkl_path = (
        '../yueyuecome/data/csi_data.pkl' if is_windows else
        '../vae/data/csi_data.pkl'
    )
    def __init__(self):
        self.qlib_data_path = DataArgs.default_qlib_data_path
        self.dataset_path = DataArgs.data_pkl_path
        self.freq = 'day'
        self.save_dir = './best_models'
        # CSI300, CSI800,CSI1000
        self.market = "csi300"
        # data split args
        # "2014-01-01" for csi1000 , "2008-01-01" ,for CSI300 or 800
        self.data_start_time = "2008-01-01"
        # "2015-01-01" for csi1000 "2009-01-01" for CSI300 or 800
        self.fit_start_time = "2008-01-01"
        self.fit_end_time = "2020-03-31"
        self.val_start_time = '2020-04-01'
        self.val_end_time = '2020-06-30'
        self.test_start_time = '2020-07-01'
        self.data_end_time = '2022-12-31'
        self.target_period = 5



def prepare_data(dataArgs):
    market_info = pd.read_csv('./data/csi_market_information.csv', encoding='utf8', header=2, index_col=0)
    # create dataloaders
    dataset_df = pd.read_pickle(dataArgs.dataset_path)
    dataset_df.rename(columns={dataset_df.columns[-1]: 'LABEL0'}, inplace=True)  # 将数据集的最后一列重命名为 'LABEL0'，表示预测因子目标。
    print("列数值区间")
    print(dataset_df.describe())
    # 确保 market_info 的索引是 datetime 类型
    market_info.index = pd.to_datetime(market_info.index)
    # 确保 dataset 的索引是 (datetime, instrument) 复合索引
    dataset_df.index = pd.MultiIndex.from_tuples(dataset_df.index, names=['datetime', 'instrument'])
    # 将 market_info 的数据插入到 dataset 的倒数第二列和最后一列之间
    # 先将 market_info 重置索引，以便进行合并
    market_info_reset = market_info.reset_index().rename(columns={'index': 'datetime'})
    # 合并数据
    merged_dataset = dataset_df.reset_index().merge(market_info_reset, on='datetime', how='left')
    # 重新设置索引
    merged_dataset.set_index(['datetime', 'instrument'], inplace=True)
    # 将"LABEL0" 列移动到最后一列
    columns = [col for col in merged_dataset.columns if col != 'LABEL0'] + ['LABEL0']
    merged_dataset = merged_dataset[columns]
    # 使用日期索引过滤数据
    dl_train_df = merged_dataset.loc[pd.IndexSlice[dataArgs.fit_start_time:dataArgs.fit_end_time, :], :]
    dl_valid_df = merged_dataset.loc[pd.IndexSlice[dataArgs.val_start_time:dataArgs.val_end_time, :], :]
    dl_test_df = merged_dataset.loc[pd.IndexSlice[dataArgs.test_start_time:dataArgs.data_end_time, :], :]
    # 打印结果
    print(dl_train_df.head())
    dl_valid = PandasDataset(dl_valid_df)
    print("验证数据转换完毕")
    dl_train = PandasDataset(dl_train_df)
    print("训练数据转换完毕")
    dl_test = PandasDataset(dl_test_df)
    return dl_train, dl_valid, dl_test



import torch
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.data = df.reset_index().to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        datetime_str = record['datetime']
        instrument = record['instrument']
        value = record['value']

        # 将 datetime 转换为 PyTorch 张量
        datetime_tensor = torch.tensor(pd.to_datetime(datetime_str).value, dtype=torch.int64)
        instrument_tensor = torch.tensor(ord(instrument), dtype=torch.int64)  # 将 instrument 转换为 ASCII 码
        value_tensor = torch.tensor(value, dtype=torch.float32)

        return datetime_tensor, instrument_tensor, value_tensor


def main():
    dataArgs = DataArgs()
    universe = dataArgs.market
    # 准备数据torch Dataset实例
    dl_train, dl_valid, dl_test=prepare_data(dataArgs)


    # # Please install qlib first before load the data.
    # with open(f'data/{universe}/{universe}_dl_train.pkl', 'rb') as f:
    #     dl_train = pickle.load(f)
    # with open(f'data/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
    #     dl_valid = pickle.load(f)
    # with open(f'data/{universe}/{universe}_dl_test.pkl', 'rb') as f:
    #     dl_test = pickle.load(f)
    # print("Data Loaded.")
    d_feat = 158
    d_model = 256
    t_nhead = 4
    s_nhead = 2
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221
    if universe == 'csi300':
        beta = 10
    elif universe == 'csi800':
        beta = 5
    n_epoch = 40
    lr = 8e-6
    GPU = 0
    seed = 0
    train_stop_loss_thred = 0.95
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout,
        S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
        save_path='model/', save_prefix=universe
    )
    # Train
    print("开始训练.....")
    model.fit(dl_train, dl_valid)
    print("Model Trained.")
    # Test
    predictions, metrics = model.predict(dl_test)
    print(metrics)
    # Load and Test
    # param_path = f'model/{universe}master_0.pkl.'
    # print(f'Model Loaded from {param_path}')
    # model.load_param(param_path)
    # predictions, metrics = model.predict(dl_test)
    # print(metrics)


main()




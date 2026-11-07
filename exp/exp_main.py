from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import csv
import os
import importlib
import pkgutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')


def build_model_dict():
    import models as _models_pkg
    model_dict = {}
    pkg_path = _models_pkg.__path__

    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if is_pkg:
            continue
        try:
            mod = importlib.import_module(f'models.{mod_name}')
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if isinstance(cls, type) and attr.startswith('Model'):
                    if attr == 'Model':
                        model_dict[mod_name] = cls
                    else:
                        model_dict[f"{mod_name}_{attr}"] = cls
        except Exception as e:
            pass

    return model_dict


MODEL_DICT = build_model_dict()
NEW_MODELS = set(MODEL_DICT.keys())


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.use_orthogonal = args.use_orthogonal

    def _build_model(self):
        if self.args.model not in MODEL_DICT:
            raise ValueError(
                f"Model '{self.args.model}' not found in models/.\n"
                f"Available: {sorted(MODEL_DICT.keys())}"
            )

        model = MODEL_DICT[self.args.model](self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        self.total_params = total_params
        print(f"[{self.args.model}] params: {total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        loss_map = {
            'mae':    nn.L1Loss(),
            'mse':    nn.MSELoss(),
            'smooth': nn.SmoothL1Loss(),
        }
        return loss_map.get(self.args.loss, nn.MSELoss())

    def _forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.model in NEW_MODELS:
            if self.use_orthogonal:
                return self.model(batch_x), True
            else:
                return self.model(batch_x), False
        else:
            if self.args.output_attention:
                out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            return out, False

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                result, _ = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(result, tuple):
                    outputs = result[0]
                else:
                    outputs = result

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val')
        test_data, test_loader   = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps   = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim   = self._select_optimizer()
        criterion     = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        max_memory = -1
        for epoch in range(self.args.train_epochs):
            iter_count  = 0
            train_loss  = []
            self.model.train()
            epoch_time  = time.time()
            time_now    = time.time()
            print(f"\n========== Epoch {epoch+1}/{self.args.train_epochs} ==========")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                result, has_ortho = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(result, tuple):
                    outputs, orthogonal_loss = result
                else:
                    outputs, orthogonal_loss = result, 0

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss    = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                back_loss = loss + getattr(self.args, 'orthogonal_weight', 0.0) * orthogonal_loss
                back_loss.backward()
                model_optim.step()

                max_memory = max(max_memory,
                                 torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Max Memory (MB): {max_memory}")
            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.2f}s")

            train_loss = np.average(train_loss)
            vali_loss  = self.vali(vali_data, vali_loader, criterion)
            test_loss  = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | "
                  f"Train: {train_loss:.7f}  Vali: {vali_loss:.7f}  Test: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'LR -> {scheduler.get_last_lr()[0]}')

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        print(f"Final Max Memory (MB): {max_memory}")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                           map_location='cuda:0'))

        preds, trues, inputx = [], [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                result, _ = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = result[0] if isinstance(result, tuple) else result

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt  = np.concatenate((inp[0, :, -1], batch_y[0, :, -1]))
                    pd  = np.concatenate((inp[0, :, -1], outputs[0, :, -1]))
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds  = np.concatenate(preds,  axis=0).reshape(-1, preds[0].shape[-2],  preds[0].shape[-1])
        trues  = np.concatenate(trues,  axis=0).reshape(-1, trues[0].shape[-2],  trues[0].shape[-1])
        inputx = np.concatenate(inputx, axis=0).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])

        os.makedirs('./results/' + setting + '/', exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rse:{rse}')

        csv_path   = './results/results.csv'
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['setting','model','params','seq_len','pred_len','mse','mae','rse'])
            writer.writerow([setting, self.args.model, self.total_params,
                             self.args.seq_len, self.args.pred_len, mse, mae, rse])

    def predict(self, setting, load=False):
        return

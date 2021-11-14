
import copy
import implicit
import numpy as np


# ---| Метрика |---
def pres_at_k(target, predict, k):
    ri = [t==p for (i, (t,p)) in enumerate(zip(target, predict)) if i<k]
    return sum(ri) / k


def average_pres_at20(target, predict):
    r = [t==p for (t,p) in zip(target, predict)]
    pk = [pres_at_k(target, predict, k)*r[k-1] for k in range(1, 21)]
    return sum(pk) / 20


def map20(targets, predicts):
    ap_i = [average_pres_at20(ti, pi) for (ti,pi) in zip(targets, predicts)]
    return sum(ap_i) / len(targets)


# ---| Модель I2I |---
class I2I():
    def __init__(self, target_ids, code_in, code_out,
                 i2i_K=101):
        # данные:
        self.target_ids = target_ids  # список целевых товаров
        self.code_in = code_in  # прямое кодирование
        self.code_out = code_out  # обратное кодирование
        # модель Item2Item:
        self.model = implicit.nearest_neighbours.CosineRecommender(K=i2i_K)

    def fit(self, X_ch):
        # X_ch - sparse matrix for I2I (транзакции, сгруппированные по чекам)
        self.model.fit(X_ch.T)  # Item2Item
        # BaseLine:
        total = np.array(X_ch.sum(axis=0)).flatten()
        sorted_all = np.vectorize(self.code_out.get)(total.argsort())
        sorted_target = sorted_all[np.isin(sorted_all, self.target_ids)]
        self.best20 = np.flip(sorted_target[-20:])

    def predict(self, x):
        # x - строка разреженной матрицы
        # предсказание:
        predi = self.model.recommend(0, x, N=974,
                                     filter_already_liked_items=False,
                                     recalculate_user=True)
        # раскодированные предсказания:
        predi_all = np.array([self.code_out[pr[0]] for pr in predi])
        # целевые товары:
        predi_targ = predi_all[np.isin(predi_all, self.target_ids)]
        # товары, не содержащиеся в чеке:
        predi_targ = predi_targ[~np.isin(predi_targ, x.nonzero()[1])]
        # если целевых рекомендованных меньше 20, то оставшиеся
        # заполним наиболее популярными среди всех клиентов:
        if predi_targ.shape[0] < 20:
            unique_b20 = self.best20[~np.isin(self.best20, predi_targ)]
            resi = np.hstack([predi_targ, unique_b20])[:20]
        else:
            resi = predi_targ[:20].copy()
        return list(resi)

    def metric(self, Y, ts):
        # Y - размеченные данные в формате, получаемом из JSON файлов
        # ts - target_sparse
        data_for_pred = copy.deepcopy(Y)
        # построение прогнозов:
        for row_id, client in enumerate(data_for_pred['clients']):
            client['predict'] = self.predict(ts[row_id])
        # расчет метрики:
        targets, predicts = [], []
        for client in data_for_pred['clients']:
            targets.append(client['target'])
            predicts.append(client['predict'])
        return map20(targets, predicts)
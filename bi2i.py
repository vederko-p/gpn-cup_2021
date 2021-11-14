
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


# ---| Вспомогательная функция объединения наборов товаров с весами |---
def my_extend(lst1, lst2):
    # unpack:
    v1, v2 = [], []
    w1, w2 = [], []
    [(v1.append(v),w1.append(w)) for v,w in lst1]  # lst1
    [(v2.append(v),w2.append(w)) for v,w in lst2]  # lst2
    # extend:
    for i in range(len(lst1)):  # перебираем все значения в списке 1
        try:
            indx = v2.index(v1[i])  # если во втором есть такое же значение
            w1[i] = (w1[i] + w2[indx]) / 2  # то берем полусумму весов
            del v2[indx]  # удаляем значение из второго списка
            del w2[indx]  # удаляем вес из второго списка
        except ValueError:
            continue
    # оставшиеся значение во втором списке уникальны, поэтому объединяем:
    v1.extend(v2)
    w1.extend(w2)
    return list(zip(v1, w1))


# ---| Модель BI2I |---
class BI2I():
    def __init__(self, cl_in_base_ids, target_ids, code_in, code_out,
                 i2i_K=41,
                 bpr_factors=32, bpr_regularization=0.06, bpr_iterations=250, bpr_random_state=2021):
        # данные:
        self.clients_in_base = cl_in_base_ids  # список id клиентов в базе
        self.target_ids = target_ids  # список целевых товаров
        self.code_in = code_in  # прямое кодирование
        self.code_out = code_out  # обратное кодирование
        # модели:
        self.model_i2i = implicit.nearest_neighbours.CosineRecommender(K=i2i_K)
        self.model_bpr = implicit.bpr.BayesianPersonalizedRanking(factors=bpr_factors,
                                                                  regularization=bpr_regularization,
                                                                  iterations=bpr_iterations,
                                                                  random_state=bpr_random_state)

    def fit(self, X_cl, X_ch):
        # X_cl - sparse matrix for BPR (транзакции, сгруппированные по клиентам)
        # X_ch - sparse matrix for I2I (транзакции, сгруппированные по чекам)
        self.X_cl = X_cl
        self.X_ch = X_ch
        print('Модель Байеса:')
        self.model_bpr.fit(X_cl.T)  # модель Байеса
        print('модель Item2Item:')
        self.model_i2i.fit(X_ch.T)  # модель Item2Item
        # BaseLine:
        total = np.array(X_ch.sum(axis=0)).flatten()
        sorted_all = np.vectorize(self.code_out.get)(total.argsort())
        sorted_target = sorted_all[np.isin(sorted_all, self.target_ids)]
        self.best20 = np.flip(sorted_target[-20:])

    def predict(self, x, cl_id, a):
        # x - строка разреженной матрицы
        # cl_id - id клиента
        # a - коэффициент предпочтения рекомендаций по модели Байеса
        # ---| 1. Предсказание по модели Item2Item |---
        pred_i2i = self.model_i2i.recommend(0, x, N=974,
                                            filter_already_liked_items=False,
                                            recalculate_user=True)
        # отсеивание нецелевых товаров:
        pred_i2i_targ = [(p[0], p[1]) for p in pred_i2i if self.code_out[p[0]] in self.target_ids]
        # отсеивание товаров, содержащихся в чеке:
        pred_i2i_targ = [(self.code_out[p[0]], p[1]) for p in pred_i2i_targ if not p[0] in x.nonzero()[1]]
        # нормировка (степень предпочтения -> вес):
        norm = sum([p[1] for p in pred_i2i_targ])
        rec_i2i = [(r[0], r[1] / norm) for r in pred_i2i_targ]  # товары и веса
        # ---| 2. Предсказание по модели Байеса |---
        try:  # клиент есть в базе:
            # номер строки в разреженной матрице (Клиент-Товар):
            row_id = self.clients_in_base.index(cl_id)
            pred_bpr = self.model_bpr.recommend(row_id, self.X_cl, N=974, filter_already_liked_items=False)
            # отсеивание нецелевых товаров:
            pred_bpr_targ = [(pb[0], pb[1]) for pb in pred_bpr if self.code_out[pb[0]] in self.target_ids]
            # отсеивание товаров, содержащихся в чеке:
            pred_bpr_targ = [(self.code_out[rb[0]], rb[1]) for rb in pred_bpr_targ if not rb[0] in x.nonzero()[1]]
            norm = sum([rb[1] for rb in pred_bpr_targ])  # нормировка (степень предпочтения -> вес)
            rec_bpr = [(rb[0], rb[1] / norm) for rb in pred_bpr_targ]  # товары и веса
            # перенормируем все полученные веса с учетом параметра альфа:
            rec_bpr = [(rb[0], rb[1] * a) for rb in rec_bpr]
            rec_i2i = [(r[0], r[1] * (1 - a)) for r in rec_i2i]
            # объединим все рекомендации и нормированные веса:
            rec_all = my_extend(rec_bpr, rec_i2i)
            # еще раз перенормируем все полученные веса:
            norm = sum([ra[1] for ra in rec_all])
            rec_all = [(ra[0], ra[1] / norm) for ra in rec_all]
            # упорядочим рекомендации по получившимся весам:
            rec_all = sorted(rec_all, key=lambda x: x[1], reverse=True)
            # вытащим только номера товаров:
            rec_all = [ra[0] for ra in rec_all]
        except ValueError:  # клиента нет в базе:
            rec_all = rec_i2i.copy()
            # упорядочим рекомендации по получившимся весам:
            rec_all = sorted(rec_all, key=lambda x: x[1], reverse=True)
            # вытащим только номера товаров:
            rec_all = [ra[0] for ra in rec_all]
        # проверим количество товаров в получившейся рекомендации; если
        # их количество < 20, то дополним рекомендацию товарами из BaseLine:
        rec_len = len(rec_all)
        if rec_len < 20:
            best20_unique = list(self.best20[~np.isin(self.best20, rec_all)])
            rec_all.extend(best20_unique[-(20 - rec_len):])
        return rec_all[:20]

    def metric(self, Y, ts, a):
        # Y - размеченные данные в формате, получаемом из JSON файлов
        # ts - target_sparse
        # a - коэффициент предпочтения рекомендаций по модели Байеса
        data_for_pred = copy.deepcopy(Y)
        # построение прогнозов:
        for row_id, client in enumerate(data_for_pred['clients']):
            client['predict'] = self.predict(ts[row_id], client['client_id'], a)
        # расчет метрики:
        targets, predicts = [], []
        for client in data_for_pred['clients']:
            targets.append(client['target'])
            predicts.append(client['predict'])
        return map20(targets, predicts)

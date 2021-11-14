
import numpy as np
import os
from tqdm.notebook import tqdm
from math import ceil
import json
from scipy import sparse
from collections import defaultdict


def get_json(grouped_data, groups_names, foldername, n=20, k=None):
    # создадим отдельную директорию, если такой еще нет:
    if not 'json_data_'+foldername in os.listdir():
        os.mkdir('json_data_'+foldername)
    if k is None:
        k = n
    groups_amount = len(groups_names)
    batch_size = ceil(groups_amount / n)
    g_id = 0
    c = 0
    if not k is None:
        n = k
    # пройдем по каждому пакету:
    for bn in tqdm(range(k)):
        group_of_clients = {'group_id':g_id, 'clients':[]}
        # пройдем по каждому клиенту в пакете:
        for gn in tqdm(groups_names[bn*batch_size: bn*batch_size+batch_size]):
            group = grouped_data.get_group(gn)
            # соберем клиента:
            client = {'client_global_id': c,
                      'cheque_id': int(group['cheque_id'].values[0]),
                      'sku_ids': [sku for sku in group['sku_id']]}
            # ID клиента:
            try:
                client['client_id'] = int(group['client_id'].values[0])
            except ValueError:
                client['client_id'] = -1
            group_of_clients['clients'].append(client)
            c += 1
        # сохраним пакет в файл:
        file_name = 'json_data_'+foldername + '/' + 'group_' + str(g_id)
        with open(file_name+'.json', 'w') as write_file:
            json.dump(group_of_clients, write_file, indent=4)
        g_id += 1
    return 0


def get_sparse_matrix(path, code_in):
    # отсортируем файлы с клиентами, чтобы не потерять порядок клиентов:
    json_names = sorted(os.listdir(path),
                        key=lambda x: int(x[6:].replace('.json', '')))
    goods_len = len(code_in.keys())
    indx_row_list, indx_col_list = [], []  # списки индексов по строке и столбцу
    values_list = []  # список значений
    # ID клиента и номер чека:
    clients_ids = []  # ID клиентов
    cheqs_ids = []  # ID чеков
    for js_name in tqdm(json_names):  # пройдем по каждому файлу
        file_path = path + '/' + js_name
        with open(file_path, 'r') as read_file:
            group_of_clients = json.load(read_file)
            for client in group_of_clients['clients']:  # по каждому клиенту
                # ID клиента и номер чека:
                clients_ids.append(client['client_id'])
                cheqs_ids.append(client['cheque_id'])
                # значения и индексы матрицы:
                indxs = defaultdict(int)
                for val in client['sku_ids']:  # сформируем покупки клиента
                    indxs[code_in[val]] += 1 / goods_len  # + нормировка
                for i in indxs:  # соберем индексы и значения для матрицы
                    indx_col_list.append(i)
                    values_list.append(indxs[i])
                indx_row_list.extend([client['client_global_id']]*len(indxs))
    # сформируем разреженную матрицу:
    sparse_matrix = sparse.coo_matrix(
        (np.array(values_list, dtype=np.float32), (indx_row_list, indx_col_list)),
        shape=(client['client_global_id']+1, goods_len))
    return sparse_matrix.tocsr(), clients_ids, cheqs_ids


def get_sparse_matrix_from_target(path, code_in):
    with open(path, 'r') as read_file:
        target_group = json.load(read_file)
        indx_row_list, indx_col_list = [], []
        values_list = []
        # ID клиента и номер чека:
        clients_ids = []  # ID клиентов
        cheqs_ids = []  # ID чеков
        c = 0
        for client in target_group['clients']:  # пройдем по всем клиентам
            # ID клиента и номер чека:
            clients_ids.append(client['client_id'])
            cheqs_ids.append(client['cheque_id'])
            for indx in client['sku_ids']:
                indx_col_list.append(code_in[int(indx)])  # индексы по столбцам
                values_list.append(client['sku_ids'][indx])  # значения
            indx_row_list.extend([c]*len(client['sku_ids'].keys()))  # индексы по строкам
            c += 1
        sparse_matrix = sparse.coo_matrix(
            (np.array(values_list, dtype=np.float32), (indx_row_list, indx_col_list)),
            shape=(c, len(code_in.keys())))
        return sparse_matrix.tocsr(), clients_ids, cheqs_ids

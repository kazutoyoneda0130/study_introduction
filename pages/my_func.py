import pulp
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
import pandas as pd
import grblogtools as glt
import shutil
import os
import time

from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) # type: ignore

class CreateRideShareProblemInstance:

    def __init__(self) -> None:
        self.car_num: int = 0
        self.car_cap: int = 0
        self.req_num: int = 0
        self.seed: int = 0
        self.form: str = ""
        self.req_node_list = []
        self.car_node_list = []
        self.coordinates_dict = {}
        self.w_dist = {}

    def set_data(self, car_num, car_cap, req_num, seed, form) -> None:
        self.car_num = car_num
        self.car_cap = car_cap
        self.req_num = req_num
        self.seed = seed
        self.form = form

    def create(self, randrange=(0, 100)) -> None:
        # 再現性確保
        random.seed(self.seed)

        # ノードリスト作成
        self.req_node_list = [f'r{i}_s' for i in range(self.req_num)] + \
                             [f'r{i}_t' for i in range(self.req_num)]
        self.car_node_list = [f'd{i}' for i in range(self.car_num)]

        # リクエストのピックアップ場所・ドロップオフ場所及び車の初期位置の座標生成
        self.coordinates_dict = {node: (random.randint(randrange[0], randrange[1]), random.randint(randrange[0], randrange[1]))
                                 for node in self.req_node_list + self.car_node_list}
        # 二点間の距離を辞書型で保持
        self.w_dist = {(s_name, t_name): np.linalg.norm(np.array(s_coordinate) - np.array(t_coordinate))
               for s_name, s_coordinate in self.coordinates_dict.items()
               for t_name, t_coordinate in self.coordinates_dict.items()
               if s_name != t_name}

    def display(self):
        self.G=nx.DiGraph()
        self.node_color = []
        for node_name,_ in self.coordinates_dict.items():
            self.G.add_node(node_name)
            if node_name[0] == 'r' and node_name[-1] == 's':
                self.node_color.append('skyblue')
            if node_name[0] == 'r' and node_name[-1] == 't':
                self.node_color.append('lightgreen')
            if node_name[0] == 'd':
                self.node_color.append('lightpink')
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw(self.G, pos=self.coordinates_dict, with_labels = True, node_size=300, font_size=8, node_color=self.node_color)
        return self.G,self.coordinates_dict,self.node_color


def g1_dict(car_cap:int,coordinates_dict:dict,w_dist:dict) -> dict:
    # 各車の対応可能リクエスト数を要素に持つリスト
    carreq_list = [c_name +'_'+ str(cap) for c_name in coordinates_dict.keys() if c_name[0] == 'd' for cap in range(car_cap)]
    # リクエストのリスト
    request_list = ['r'+re.findall(r'\d+', request)[0] for request in coordinates_dict.keys() if request[0] == 'r' and request[-1] == 's']
    # 辞書G1_weight_dict　キー：(left vertex-set, right vertex-set)、バリュー：二点間の距離
    G1_weight_dict = {}
    for carreq in carreq_list:
        for req in request_list:
            if int(re.findall(r'\d+', carreq)[1]) < car_cap-1:
                G1_weight_dict[(carreq, req)] = w_dist[('d'+re.findall(r'\d+', carreq)[0],req+'_s')] + w_dist[(req+'_s',req+'_t')] + w_dist[(req+'_t','d'+re.findall(r'\d+', carreq)[0])]
            else:
                G1_weight_dict[(carreq, req)] = w_dist[('d'+re.findall(r'\d+', carreq)[0],req+'_s')] + w_dist[(req+'_s',req+'_t')]
    return G1_weight_dict

def g2_dict(car_cap:int,coordinates_dict:dict,w_dist:dict) -> dict:
    # 各車の対応可能リクエスト数を要素に持つリスト
    carreq_list = [c_name +'_'+ str(cap) for c_name in coordinates_dict.keys() if c_name[0] == 'd' for cap in range(car_cap)]
    # リクエストのリスト
    request_list = ['r'+re.findall(r'\d+', request)[0] for request in coordinates_dict.keys() if request[0] == 'r' and request[-1] == 's']
    # 辞書G1_weight_dict　キー：(left vertex-set, right vertex-set)、バリュー：二点間の距離
    G2_weight_dict = {}
    for carreq in carreq_list:
        for req in request_list:
            if int(re.findall(r'\d+', carreq)[1]) < car_cap-1:
                G2_weight_dict[(carreq, req)] = (car_cap - 1 - int(re.findall(r'\d+', carreq)[1]) + 1)*(w_dist[('d'+re.findall(r'\d+', carreq)[0],req+'_s')] + w_dist[(req+'_s',req+'_t')]) + (car_cap - 1 - int(re.findall(r'\d+', carreq)[1]))*w_dist[(req+'_t','d'+re.findall(r'\d+', carreq)[0])]
            else:
                G2_weight_dict[(carreq, req)] = w_dist[('d'+re.findall(r'\d+', carreq)[0],req+'_s')] + w_dist[(req+'_s',req+'_t')]
    return G2_weight_dict

def culc_cslat(G:nx.classes.graph.Graph,car_num:int,car_cap:int,w_dist:dict):
    path_dict = {f'd{car}':[f'd{car}'] + [node for node in nx.all_neighbors(G,f'd{car}')] for car in range(car_num)}
    for car in range(car_num):
        for _ in range(2 * car_cap - 1):
            path_dict[f'd{car}'] += G.successors(path_dict[f'd{car}'][-1]) # type: ignore
    cslat = 0
    for _,route_list in path_dict.items():
        for route_index,node in enumerate(route_list):
            if node[-1] == 't':
                for k,v in enumerate(route_list[:route_index+1]):
                    if k != len(route_list[:route_index+1]) - 1:
                        cslat += w_dist[(route_list[:route_index+1][k], route_list[:route_index+1][k+1])]
    return cslat

class find_bipartite_graph_min_weight_assignment:

    # 初期化
    def __init__(self) -> None:
        self.left_ver_set:list
        self.right_ver_set:list
        self.weight_dict:dict

    # データのセット
    def set_data(self,weight_dict,algorithm) -> None:
        self.left_ver_set = list(set([k[0] for k in weight_dict.keys()]))
        self.right_ver_set = list(set([k[1] for k in weight_dict.keys()]))
        self.weight_dict = weight_dict
        self.algorithm = algorithm

    # 数理モデルの構築
    def build_model(self) -> None:
        ### 数理モデルの定義　###
        self.model = pulp.LpProblem("bipartite_graph_min_weight_assignment", pulp.LpMinimize)

        ### 変数の定義 ###
        # 集合Aと集合Bの要素をマッチングするなら1、しないなら0
        self.x = pulp.LpVariable.dicts("x", [k for k in self.weight_dict.keys()], cat="Binary")

        ### 目的関数の定義 ###
        self.model += pulp.lpSum([self.x[lr_ver_set]*weight for lr_ver_set,weight in self.weight_dict.items()])

        ### 制約式の定義 ###
        # 集合Aに属する要素は集合Bの要素のうちただ一つとマッチングする
        for left_ver in self.left_ver_set:
            self.model += pulp.lpSum([self.x[(left_ver,right_ver)] for right_ver in self.right_ver_set]) == 1
        # 集合Bに属する要素は集合Aの要素のうちただ一つとマッチングする
        for right_ver in self.right_ver_set:
            self.model += pulp.lpSum([self.x[(left_ver,right_ver)] for left_ver in self.left_ver_set]) == 1

    # 数理モデルの実行
    def solve(self) -> None:
        #solver = pulp.PULP_CBC_CMD(msg=False)
        solver = pulp.GUROBI(msg=False)
        self.status = self.model.solve(solver)
        self.status = pulp.LpStatus[self.status]
        self.objective = pulp.value(self.model.objective)
        if self.algorithm == 'TA':
            self.selected_arc_dict = {'d'+re.findall(r'\d+', v.name)[0]+'_'+re.findall(r'\d+', v.name)[1]:'r'+re.findall(r'\d+', v.name)[2] for v in self.model.variables() if v.varValue == 1}
        if self.algorithm == 'MA':
            self.selected_arc_dict = {'d'+re.findall(r'\d+', v.name)[0]:('r'+re.findall(r'\d+', v.name)[1],'r'+re.findall(r'\d+', v.name)[2]) for v in self.model.variables() if v.varValue == 1}

def display(selected_arc_dict:dict,coordinates_dict:dict,color_list:list,request_num:int,w_dist:dict,algorithm:str) -> list:
    G=nx.DiGraph()
    pos = {}
    node_color = []
    add_edges_list = []
    if algorithm == "TA":
        for car,req in selected_arc_dict.items():
            car_number, served_order = re.findall(r'\d+', car)
            add_edges_list.append((req+'_s',req+'_t',{'color' : color_list[int(car_number)]}))
            if served_order == "0":
                add_edges_list.append(('d'+car_number,req+'_s',{'color' : color_list[int(car_number)]}))
            else:
                add_edges_list.append((selected_arc_dict['d'+car_number+'_'+str(int(served_order)-1)]+'_t',req+'_s',{'color' : color_list[int(car_number)]}))

    u_dicta = u_dict(request_num=request_num,w_dist=w_dist)
    if algorithm == "MA":
        for car,req in selected_arc_dict.items():
            car_number = re.findall(r'\d+', car)[0]
            add_edges_list.append((car,f'{req[0]}_s',{'color' : color_list[int(car_number)]}))
            if u_dicta[req] == w_dist[(f'{req[0]}_s',f'{req[1]}_s')] + w_dist[(f'{req[1]}_s',f'{req[0]}_t')] + w_dist[(f'{req[0]}_t',f'{req[1]}_t')]:
                add_edges_list += [(f'{req[0]}_s',f'{req[1]}_s',{'color' : color_list[int(car_number)]}),
                                   (f'{req[1]}_s',f'{req[0]}_t',{'color' : color_list[int(car_number)]}),
                                   (f'{req[0]}_t',f'{req[1]}_t',{'color' : color_list[int(car_number)]})]
            if u_dicta[req] == w_dist[(f'{req[0]}_s',f'{req[1]}_s')] + w_dist[(f'{req[1]}_s',f'{req[1]}_t')] + w_dist[(f'{req[1]}_t',f'{req[0]}_t')]:
                add_edges_list += [(f'{req[0]}_s',f'{req[1]}_s',{'color' : color_list[int(car_number)]}),
                                   (f'{req[1]}_s',f'{req[1]}_t',{'color' : color_list[int(car_number)]}),
                                   (f'{req[1]}_t',f'{req[0]}_t',{'color' : color_list[int(car_number)]})]
            if u_dicta[req] == w_dist[(f'{req[0]}_s',f'{req[0]}_t')] + w_dist[(f'{req[0]}_t',f'{req[1]}_s')] + w_dist[(f'{req[1]}_s',f'{req[1]}_t')]:
                add_edges_list += [(f'{req[0]}_s',f'{req[0]}_t',{'color' : color_list[int(car_number)]}),
                                   (f'{req[0]}_t',f'{req[1]}_s',{'color' : color_list[int(car_number)]}),
                                   (f'{req[1]}_s',f'{req[1]}_t',{'color' : color_list[int(car_number)]})]

    for node_name,coordinates in coordinates_dict.items():
        G.add_node(node_name)
        pos[node_name] = coordinates
        if node_name[0] == 'r' and node_name[-1] == 's':
            node_color.append('skyblue')
        if node_name[0] == 'r' and node_name[-1] == 't':
            node_color.append('lightgreen')
        if node_name[0] == 'd':
            node_color.append('lightpink')

    G.add_edges_from(add_edges_list)

    edge_color = [edge['color'] for edge in G.edges.values()]

    return [edge_color,G,[edge[:2] for edge in add_edges_list]]

class find_one_graph_min_weight_assignment:

    # 初期化
    def __init__(self) -> None:
        self.ver_set:list
        self.weight_dict:dict

    # データのセット
    def set_data(self,weight_dict) -> None:
        self.ver_set = list(set([k[0] for k in weight_dict.keys()] + [k[1] for k in weight_dict.keys()]))
        self.weight_dict = weight_dict

    def build_model(self):
        ### 数理モデルの定義　###
        self.model = pulp.LpProblem("one_graph_min_weight_assignment", pulp.LpMinimize)
        ### 変数の定義 ###
        x = pulp.LpVariable.dicts("x", [k for k in self.weight_dict.keys()], cat="Binary")
        self.model += pulp.lpSum([x[lr_ver_set]*weight for lr_ver_set,weight in self.weight_dict.items()])

        for i in range(len(self.ver_set)):
            self.model += pulp.lpSum([x[r_pair] for r_pair in self.weight_dict.keys() if re.findall(r'\d+', r_pair[0])[0] == str(i) or re.findall(r'\d+', r_pair[1])[0] == str(i)]) == 1

    # 数理モデルの実行
    def solve(self) -> None:
        #solver = pulp.PULP_CBC_CMD(msg=False)
        solver = pulp.GUROBI(msg=False)
        self.status = self.model.solve(solver)
        self.status = pulp.LpStatus[self.status]
        self.objective = pulp.value(self.model.objective)
        self.selected_arc_dict = ['r'+re.findall(r'\d+', v.name)[0] + '_' + 'r'+re.findall(r'\d+', v.name)[1] for v in self.model.variables() if v.varValue == 1]

def u_dict(request_num:int,w_dist:dict):
    u_dict = {}
    for r_s in range(request_num):
        for r_t in range(request_num):
            if r_s != r_t:
                u_dict[(f'r{r_s}',f'r{r_t}')] = min(w_dist[(f'r{r_s}_s',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_s}_t')] + w_dist[(f'r{r_s}_t',f'r{r_t}_t')],
                                                    w_dist[(f'r{r_s}_s',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_t}_t')] + w_dist[(f'r{r_t}_t',f'r{r_s}_t')],
                                                    w_dist[(f'r{r_s}_s',f'r{r_s}_t')] + w_dist[(f'r{r_s}_t',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_t}_t')])

    return u_dict

def m_dict(request_num:int,w_dist:dict):
    m_dict = {}
    for r_s in range(request_num):
        for r_t in range(request_num):
            if r_s != r_t:
                m_dict[(f'r{r_s}',f'r{r_t}')] = min(2*(w_dist[(f'r{r_s}_s',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_s}_t')]) + w_dist[(f'r{r_s}_t',f'r{r_t}_t')],
                                                    2*(w_dist[(f'r{r_s}_s',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_t}_t')]) + w_dist[(f'r{r_t}_t',f'r{r_s}_t')],
                                                    2*w_dist[(f'r{r_s}_s',f'r{r_s}_t')] + w_dist[(f'r{r_s}_t',f'r{r_t}_s')] + w_dist[(f'r{r_t}_s',f'r{r_t}_t')])

    return m_dict

def g3_dict(request_num:int,v_dict:dict):
    G3_weight_dict = {}
    for rf in range(request_num):
        for rs in range(request_num):
            if rf < rs:
                G3_weight_dict[(f'r{rf}',f'r{rs}')] = (v_dict[(f'r{rf}',f'r{rs}')] + v_dict[(f'r{rs}',f'r{rf}')]) / 2
    return G3_weight_dict

def g4_dict(car_num:int,req_pair_list:list,w_dist:dict,v_dict:dict,objective:str):
    if objective == 'CSsum':
        a = 1
    if objective == 'CSlat':
        a = 2
    G4_weight_dict = {}
    for req_pair in req_pair_list:
        for car in range(car_num):
            R = re.findall(r'\d+', req_pair)
            G4_weight_dict[(f'd{car}',req_pair)] = min(a * w_dist[(f'd{car}',f'r{R[0]}_s')] + ((v_dict[(f'r{R[0]}',f'r{R[1]}')] - v_dict[(f'r{R[1]}',f'r{R[0]}')]) / 2),
                                                       a * w_dist[(f'd{car}',f'r{R[1]}_s')] - ((v_dict[(f'r{R[0]}',f'r{R[1]}')] - v_dict[(f'r{R[1]}',f'r{R[0]}')]) / 2))
    return G4_weight_dict

class multicap_rideshare_prob:

    # 初期化
    def __init__(self):
        self.car_num:int
        self.car_cap:int
        self.req_num:int
        self.w_dist:dict
        self.d_uv:dict
        self.D:list
        self.R:list
        self.L:list
        self.N:list
        self.E:list
        self.coordinates_dict:dict
        self.form:str
        self.seed:int
        self.timelimit:int
        self.H:list

    # データのセット
    def set_data(self,car_num,car_cap,req_num,w_dist,coordinates_dict,seed,form="$x_{uv},z_{uk}$",timelimit=1200):
        self.car_num = car_num
        self.car_cap = car_cap
        self.req_num = req_num
        self.w_dist = w_dist
        self.d_uv = {k:0 if (k[0][-1] == 't' and k[1][0] == 'd') else v for k,v in w_dist.items()}
        self.D = [str(i) for i in range(car_num)]
        self.R = [f'r{i}' for i in range(self.req_num)]
        self.L = [f'd{i}' for i in self.D] + [f'{r}_s' for r in self.R] + [f'{r}_t' for r in self.R]
        self.N = [str(i) for i in range(len(self.L))]
        self.E = [str(i*(2*car_cap+1)) for i in range(car_num)]
        self.coordinates_dict = coordinates_dict
        self.form = form
        self.seed = seed
        self.timelimit = timelimit
        self.H = [str(i) for i in range(2*car_cap+1)]

    # セットされたデータの表示
    def show(self):
        print('=' * 50)
        print(f'車の数 : {self.car_num}')
        print(f'車のキャパシティー : {self.car_cap}')
        print(f'リクエスト数 : {self.req_num}')
        print(f'元の辞書 : {self.w_dist}')
        print(f'この問題用に作成した辞書 : {self.d_uv}')
        print(f'車の集合 : {self.D}')
        print(f'リクエストの集合 : {self.R}')
        print(f'ノードの集合 : {self.L}')
        print(f'ノードを訪ねる順番の集合 : {self.N}')
        print(f'車の初期位置に訪れる順番の集合 : {self.E}')
        print('=' * 50)

    # 数理モデルの構築
    def build_model(self):
        ### 数理モデルの定義　###
        self.model = pulp.LpProblem("multi_cap_ride_sharing_problem", pulp.LpMinimize)
        if self.form == "$x_{uv},z_{uk},a_{ri}$" or self.form == "$x_{uv},z_{uk}$":
            ### 変数の定義 ###
            # ノードuの次にノードvを訪問するなら1、しないなら0
            self.xuv = pulp.LpVariable.dicts("xuv", [k for k in self.d_uv.keys()], cat="Binary")
            self.zuk = pulp.LpVariable.dicts("zuk", [(u, k) for u in self.L for k in self.N], cat="Binary")
            ### 目的関数の定義 ###
            self.model += pulp.lpSum([self.xuv[cor]*dist for cor,dist in self.d_uv.items()])
            ### 制約式の定義 ###
            # 各ノードの前にいずれかのノードが訪問される
            for v in self.L:
                self.model += pulp.lpSum([self.xuv[(u,v)] for u in list(set(self.L) - set([v]))]) == 1
            # 各ノードの後にいずれかのノードが訪問される
            for u in self.L:
                self.model += pulp.lpSum([self.xuv[(u,v)] for v in list(set(self.L) - set([u]))]) == 1
            # 各順番においていずれかのノードが訪問される
            for k in self.N:
                self.model += pulp.lpSum([self.zuk[(u,k)] for u in self.L]) == 1
            # 各ノードはいずれかの順番で訪問される
            for u in self.L:
                self.model += pulp.lpSum([self.zuk[(u,k)] for k in self.N]) == 1
            # 車の初期位置が訪問されなければいけない順番に訪問される
            for i in list(set(self.D) - set(["0"])):
                self.model += pulp.lpSum([self.zuk[(f'd{i}',k)] for k in list(set(self.E) - set(["0"]))]) == 1
            # 車の初期位置が訪問された直後は必ずいずれかのリクエストのピックアップ場所が訪ねられる
            # 車の初期位置が訪問された2c個後(車の初期位置が訪問される1つ前)は必ずいずれかのリクエストのドロップオフ場所が訪ねられる
            for e in self.E:
                k_rs = str(int(e) + 1)
                self.model += pulp.lpSum([self.zuk[(f'{r}_s',k_rs)] for r in self.R]) == 1
                k_rt = str(int(e) + 2*self.car_cap)
                self.model += pulp.lpSum([self.zuk[(f'{r}_t',k_rt)] for r in self.R]) == 1
            # z_uk=1かつz_vk+1=1であるときに必ずx_uv=1となるため,ノードuの次にノードvが訪問されるときにx_uv=1となり,ノードuとノードv間の移動時間が考慮される
            # 最後に訪ねられる|L|番目のノードから最初に訪ねられるノードへの移動時間が考慮される
            for u in self.L:
                for v in list(set(self.L) - set([u])):
                    self.model += self.zuk[(u,self.N[-1])] + self.zuk[(v,"0")] - 1 <= self.xuv[(u,v)]
                    for k in self.N[:-1]:
                        self.model += self.zuk[(u,k)] + self.zuk[(v,str(int(k)+1))] - 1 <= self.xuv[(u,v)]
            # 各リスエストのピックアップ場所はドロップオフ場所よりも前に訪ねられ, ピックアップ場所を訪ねてから2c−1ノード以内にドロップオフ場所が訪ねられる
            for l in self.N:
                for r in self.R:
                    self.model += pulp.lpSum([self.zuk[(f"{r}_t",str(k))] - self.zuk[(f"{r}_s",str(k))] for k in range(int(l)+1)]) <= 0
                    self.model += pulp.lpSum([self.zuk[(f"{r}_t",str(k))] - self.zuk[(f"{r}_s",str(k))] for k in range(int(l)+1)]) >= -1

            # 変数xuv,zuk,ariを使った定式化
            if self.form == "$x_{uv},z_{uk},a_{ri}$":
                self.a = pulp.LpVariable.dicts("a", [(r, d) for r in self.R for d in self.D], cat="Binary")
                # 各リクエストはいずれか1つの車に割当てられる
                for r in self.R:
                    self.model += pulp.lpSum([self.a[(r,d)] for d in self.D]) == 1
                # 各車はc個のリクエストを行う
                for d in self.D:
                    self.model += pulp.lpSum([self.a[(r,d)] for r in self.R]) == self.car_cap
                # リクエストのピックアップ場所s_rを用いたzとaの関係式
                # リクエストのドロップオフ場所t_rを用いたzとaの関係式
                for e in self.E:
                    for r in self.R:
                        for d in self.D:
                            self.model += pulp.lpSum([self.zuk[(f"{r}_s",str(int(e)+i))] for i in range(2*self.car_cap)]) >= self.zuk[(f"d{d}",e)] + self.a[(r,d)] - 1
                            self.model += pulp.lpSum([self.zuk[(f"{r}_t",str(int(e)+i))] for i in range(2*self.car_cap+1)]) >= self.zuk[(f"d{d}",e)] + self.a[(r,d)] - 1
                ## 変数x,zのときのみ用いていた制約式をx,z,aのときも追加
                #for k in self.E:
                #    self.model += pulp.lpSum([pulp.lpSum([self.zuk[(f"{r}_s",str(i))] for i in range(int(k)+1,int(k) + 2 * self.car_cap)]) for r in self.R]) == self.car_cap
                #    self.model += pulp.lpSum([pulp.lpSum([self.zuk[(f"{r}_t",str(i))] for i in range(int(k)+2,int(k) + 2 * self.car_cap + 1)]) for r in self.R]) == self.car_cap

            # 変数xuv,zukを使った定式化
            if self.form == "$x_{uv},z_{uk}$":
                 for k in self.E:
                     self.model += pulp.lpSum([pulp.lpSum([self.zuk[(f"{r}_s",str(i))] for i in range(int(k)+1,int(k) + 2 * self.car_cap)]) for r in self.R]) == self.car_cap
                     self.model += pulp.lpSum([pulp.lpSum([self.zuk[(f"{r}_t",str(i))] for i in range(int(k)+2,int(k) + 2 * self.car_cap + 1)]) for r in self.R]) == self.car_cap
            # d1が一番最初に訪問される
            self.model += self.zuk[("d0","0")] == 1

        if self.form == "$x_{uv},z_{iuk}$":
            ### 変数の定義 ###
            # ノードuの次にノードvを訪問するなら1、しないなら0
            self.xuv = pulp.LpVariable.dicts("xuv", [k for k in self.d_uv.keys()], cat="Binary")
            self.ziuk = pulp.LpVariable.dicts("ziuk", [(i, u, k) for i in self.D for u in self.L for k in self.N], cat="Binary")
            ### 目的関数の定義 ###
            self.model += pulp.lpSum([self.xuv[cor]*dist for cor,dist in self.d_uv.items()])
            ### 制約式の定義 ###
            # 各ノードの前にいずれかのノードが訪問される
            for v in self.L:
                self.model += pulp.lpSum([self.xuv[(u,v)] for u in list(set(self.L) - set([v]))]) == 1
            # 各ノードの後にいずれかのノードが訪問される
            for u in self.L:
                self.model += pulp.lpSum([self.xuv[(u,v)] for v in list(set(self.L) - set([u]))]) == 1

            # 各順番においていずれかのノードが訪問される
            for k in list(set(self.H) - set(["0","1",str(2*self.car_cap)])):
                for i in self.D:
                    self.model += pulp.lpSum([self.ziuk[(i,u,k)] for u in self.L]) == 1
            # 各ノードはいずれかの順番でいずれかの車に訪問される
            for u in self.L:
                self.model += pulp.lpSum([self.ziuk[(i,u,k)] for i in self.D for k in self.H]) == 1
            # 各車はc個のリクエストのピックアップ場所を訪問する
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s",k)] for r in self.R for k in self.H]) == self.car_cap
            # 各車はc個のリクエストのドロップオフ場所を訪問する
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",k)] for r in self.R for k in self.H]) == self.car_cap
            # あるリクエストのピックアップ場所とドロップオフ場所は必ず同じ車に訪問される
            for r in self.R:
                for i in self.D:
                    self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s",k)] for k in self.H]) == pulp.lpSum([self.ziuk[(i,f"{r}_t",k)] for k in self.H])
            # 2番目には,あるリクエストのピックアップ場所が訪問される
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s","1")] for r in self.R]) == 1
            # 2c+1番目には,あるリクエストのドロップオフ場所が訪問される
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",str(2*self.car_cap))] for r in self.R]) == 1
            # z_{iuk}=1かつz_{ivk+1}=1であるときに必ずx_{uv}=1となるため,ノードuの次にノードvが訪問されるときにx_{uv}=1となり,ノードuとノードv間の移動時間が考慮される
            for u in self.L:
                for v in list(set(self.L) -set([u])):
                    for i in self.D:
                        for k in list(set(self.H) - set([str(2*self.car_cap)])):
                            self.model += self.ziuk[(i,u,k)] + self.ziuk[(i,v,str(int(k)+1))] -1 <= self.xuv[(u,v)]
            # 各リスエストのピックアップ場所はドロップオフ場所よりも前に訪ねられる
            for i in self.D:
                for l in list(set(self.H)-set(["0",str(2*self.car_cap)])):
                    for r in self.R:
                        self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",str(k))] - self.ziuk[(i,f"{r}_s",str(k))] for k in range(1,int(l)+1)]) >= -1
                        self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",str(k))] - self.ziuk[(i,f"{r}_s",str(k))] for k in range(1,int(l)+1)]) <= 0
            # 各車の初期位置は一番最初に訪問される
            for i in self.D:
                self.model += self.ziuk[(i,f"d{i}","0")] == 1

        if self.form == "$m_{Pi}$":
            ### 変数の定義 ###
            self.mPi = pulp.LpVariable.dicts("mPi", [k for k in self.w_dist.keys()], cat="Binary")
            ### 目的関数の定義 ###
            self.model += pulp.lpSum([self.mPi[cor]*dist for cor,dist in self.w_dist.items()])
            ### 制約式の定義 ###
            if self.car_cap == 2:
                for _,i in enumerate(self.D):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_s,r_f)] for p,r_s in enumerate(self.R) for q,r_f in enumerate(self.R) if p < q]) == 1
                for a,r in enumerate(self.R):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r,r_f)] for q,r_f in enumerate(self.R) for i in self.D if a < q] + [self.mPi[(f"d{i}",r_s,r)] for p,r_s in enumerate(self.R) for i in self.D if p < a]) == 1
            if self.car_cap == 3:
                for _,i in enumerate(self.D):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_s,r_f,r_l)] for p,r_s in enumerate(self.R) for q,r_f in enumerate(self.R) for r,r_l in enumerate(self.R) if p < q < r]) == 1
                for r,r_l in enumerate(self.R):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_s,r_f,r_l)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if p < q < r] +
                                             [self.mPi[(f"d{i}",r_s,r_l,r_f)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if p < r < q] +
                                             [self.mPi[(f"d{i}",r_l,r_s,r_f)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if r < p < q]) == 1

        if self.form == "no_constant_cap_$x_{uv},z_{iuk}$":
            ### 変数の定義 ###
            # ノードuの次にノードvを訪問するなら1、しないなら0
            self.xuv = pulp.LpVariable.dicts("xuv", [k for k in self.d_uv.keys()], cat="Binary")
            self.ziuk = pulp.LpVariable.dicts("ziuk", [(i, u, k) for i in self.D for u in self.L for k in self.N], cat="Binary")
            ### 目的関数の定義 ###
            self.model += pulp.lpSum([self.xuv[cor]*dist for cor,dist in self.d_uv.items()])
            ### 制約式の定義 ###
            # 各順番においていずれかのノードが訪問される
            for k in self.N:
                for i in self.D:
                    self.model += pulp.lpSum([self.ziuk[(i,u,k)] for u in self.L]) <= 1
            # 各ノードはいずれかの順番でいずれかの車に訪問される
            for u in self.L:
                self.model += pulp.lpSum([self.ziuk[(i,u,k)] for i in self.D for k in self.N]) == 1
            ## # 各ノードはいずれかの順番でいずれかの車に訪問される
            ## for u in self.L:
            ##     self.model += pulp.lpSum([self.ziuk[(i,u,k)] for i in self.D for k in self.N]) <= 1
            # 初期位置からずっと1をとり、あるところからはずっと0をとる
            for i in self.D:
                for k in self.N[:-1]:
                    self.model += pulp.lpSum([self.ziuk[(i,u,str(int(k)+1))] - self.ziuk[(i,u,k)] for u in self.L]) <= 0
            # 各車はc個のリクエストのピックアップ場所を訪問する
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s",k)] for r in self.R for k in self.N]) <= self.car_cap
            # 各車はc個のリクエストのドロップオフ場所を訪問する
            for i in self.D:
                self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",k)] for r in self.R for k in self.N]) <= self.car_cap
            # あるリクエストのピックアップ場所とドロップオフ場所は必ず同じ車に訪問される
            for r in self.R:
                for i in self.D:
                    self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s",k)] for k in self.N]) == pulp.lpSum([self.ziuk[(i,f"{r}_t",k)] for k in self.N])
            # # 2番目には,あるリクエストのピックアップ場所が訪問される
            # for i in self.D:
            #     self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s","1")] for r in self.R]) == 1
            ####     # リクエストはm個実行される
            ####     self.model += pulp.lpSum([self.ziuk[(i,f"{r}_s",k)] for i in self.D for r in self.R for k in self.N]) == self.req_num
            ####     self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",k)] for i in self.D for r in self.R for k in self.N]) == self.req_num
            # z_{iuk}=1かつz_{ivk+1}=1であるときに必ずx_{uv}=1となるため,ノードuの次にノードvが訪問されるときにx_{uv}=1となり,ノードuとノードv間の移動時間が考慮される
            for u in self.L:
                for v in set(self.L) -set([u]):
                    for i in self.D:
                        for k in self.N[:-1]:
                            self.model += self.ziuk[(i,u,k)] + self.ziuk[(i,v,str(int(k)+1))] -1 <= self.xuv[(u,v)]
            # 各リスエストのピックアップ場所はドロップオフ場所よりも前に訪ねられる
            for i in self.D:
                for l in self.N[1:-1]:
                    for r in self.R:
                        self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",str(k))] - self.ziuk[(i,f"{r}_s",str(k))] for k in range(1,int(l)+1)]) >= -1
                        self.model += pulp.lpSum([self.ziuk[(i,f"{r}_t",str(k))] - self.ziuk[(i,f"{r}_s",str(k))] for k in range(1,int(l)+1)]) <= 0
            # 各車の初期位置は一番最初に訪問される
            for i in self.D:
                self.model += self.ziuk[(i,f"d{i}","0")] == 1

        if self.form == "no_constant_cap_$m_{Pi}$":
            ### 変数の定義 ###
            self.mPi = pulp.LpVariable.dicts("mPi", [k for k in self.w_dist.keys()], cat="Binary")
            ### 目的関数の定義 ###
            self.model += pulp.lpSum([self.mPi[cor]*dist for cor,dist in self.w_dist.items()])
            ### 制約式の定義 ###
            if self.car_cap == 2:
                for _,i in enumerate(self.D):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_s)] for p,r_s in enumerate(self.R)] +
                                             [self.mPi[(f"d{i}",r_s,r_f)] for p,r_s in enumerate(self.R) for q,r_f in enumerate(self.R) if p < q]) <= 1
                for a,r in enumerate(self.R):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r)] for i in self.D] +
                                             [self.mPi[(f"d{i}",r,r_f)] for q,r_f in enumerate(self.R) for i in self.D if a < q] +
                                             [self.mPi[(f"d{i}",r_s,r)] for p,r_s in enumerate(self.R) for i in self.D if p < a]) == 1
            if self.car_cap == 3:
                for _,i in enumerate(self.D):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_s)] for p,r_s in enumerate(self.R)] +
                                             [self.mPi[(f"d{i}",r_s,r_f)] for p,r_s in enumerate(self.R) for q,r_f in enumerate(self.R) if p < q] +
                                             [self.mPi[(f"d{i}",r_s,r_f,r_l)] for p,r_s in enumerate(self.R) for q,r_f in enumerate(self.R) for r,r_l in enumerate(self.R) if p < q < r]) <= 1
                for r,r_l in enumerate(self.R):
                    self.model += pulp.lpSum([self.mPi[(f"d{i}",r_l)] for i in self.D] +
                                             [self.mPi[(f"d{i}",r_l,r_f)] for q,r_f in enumerate(self.R) for i in self.D if r < q] +
                                             [self.mPi[(f"d{i}",r_s,r_l)] for p,r_s in enumerate(self.R) for i in self.D if p < r] +
                                             [self.mPi[(f"d{i}",r_s,r_f,r_l)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if p < q < r] +
                                             [self.mPi[(f"d{i}",r_s,r_l,r_f)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if p < r < q] +
                                             [self.mPi[(f"d{i}",r_l,r_s,r_f)] for q,r_f in enumerate(self.R) for p,r_s in enumerate(self.R) for i in self.D if r < p < q]) == 1

    # 数理モデルの実行
    def solve(self):
        if self.form == "no_constant_cap_$x_{uv},z_{iuk}$" or "no_constant_cap_$m_{Pi}$":
            file_name = f"form:{self.form}_carcap:{self.car_cap}_reqnum:{self.car_num}_seed:{self.seed}.log"
        else:
            file_name = f"form:{self.form}_carcap:{self.car_cap}_carnum:{self.car_num}_seed:{self.seed}.log"
        solver = pulp.GUROBI(msg=True,logfile=file_name,timeLimit=self.timelimit)
        self.status = self.model.solve(solver) # type: ignore
        if self.form == "no_constant_cap_$x_{uv},z_{iuk}$" or "no_constant_cap_$m_{Pi}$":
            results = glt.parse([f"form:{self.form}_carcap:{self.car_cap}_reqnum:{self.car_num}_seed:{self.seed}.log"])
        else:
            results = glt.parse([f"form:{self.form}_carcap:{self.car_cap}_carnum:{self.car_num}_seed:{self.seed}.log"])

        solvetime = pd.DataFrame(results.progress("nodelog")).tail(1)["Time"].values
        with open(file_name, newline='') as text:
            result_line = re.findall(r'Presolved:(.*)',text.read())
            if len(result_line) == 0:
                constraint_num,variable_num = [0,0]
            else:
                constraint_num,variable_num = [int(i) for i in re.findall(r'\d+',result_line[0])][:-1]
        return pulp.LpStatus[self.status],pulp.value(self.model.objective),solvetime,constraint_num,variable_num

    def display(self,coordinates_dict,node_color,save_name):
        new_graph = nx.DiGraph()
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) # type: ignore
        edge_color_list = list(colors.keys())
        selected_arc_list_x = [v for v in self.model.variables() if v.varValue == 1 and v.name[0] == "x"]
        selected_arc_list = [tuple(j[1:-1]for j in re.findall(r"\'.*?\'",str(arc))) for arc in selected_arc_list_x]
        selected_arc_dict = {"d"+d : ["d"+d] for d in self.D}
        while len(selected_arc_list) > 0:
            for arc in selected_arc_list:
                for d in self.D:
                    if arc[0] in selected_arc_dict["d"+d]:
                        if arc[1][0] != "d":
                            selected_arc_dict["d"+d].append(arc[1])
                        selected_arc_list.remove(arc)
        for d in self.D:
            add_adges_list = []
            for i,arc in enumerate(selected_arc_dict["d"+d]):
                new_graph.add_node("d"+d,color="r")
                if len(selected_arc_dict["d"+d])-1 > i:
                    add_adges_list.append((selected_arc_dict["d"+d][i],selected_arc_dict["d"+d][i+1],{'color' : edge_color_list[2:][int(d)]}))
            new_graph.add_edges_from(add_adges_list)
        edge_color = [edge['color'] for edge in new_graph.edges.values()]
        c = ["g" if node_name[0]=="d" else "b" for node_name in new_graph.nodes()]
        plt.figure(figsize=(12,12))
        #nx.draw(new_graph, pos=coordinates_dict, with_labels = True, node_size=300,edge_color=edge_color,font_size=8, node_color=node_color)
        nx.draw(new_graph, pos=coordinates_dict, with_labels = True, node_size=300,edge_color=edge_color,font_size=8,node_color=c)
        plt.savefig(save_name)

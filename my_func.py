import random
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import pulp
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
            if node_name[-1] == 's':
                self.node_color.append('skyblue')
            if node_name[0] == 'r' and node_name[-1] == 't':
                self.node_color.append('lightgreen')
            if node_name[0] == 'd':
                self.node_color.append('lightpink')

class MulticapRideshareProblem:
    # 初期化
    def __init__(self) -> None:
        self.car_num:int = 0         # 車の台数
        self.car_cap:int = 0         # 各車の定員
        self.req_num:int = 0         # リクエスト数
        self.w_dist:dict = {}        # ノード間の距離（生データ）
        self.d_uv:dict  = {}         # モデルで使う距離（調整後）
        self.D:list = []             # 車集合
        self.R:list = []             # リクエスト集合（"r0", "r1", ...）
        self.L:list = []             # 全ノード集合（出発・ピックアップ・ドロップオフ）
        self.N:list = []             # ノードの訪問順序インデックス集合
        self.E:list = []             #　車の初期位置を訪れる順番の集合
        self.coordinates_dict:dict = {}  # ノードの座標
        self.form:str   = ""         # モデル定式化の形式
        self.seed:int   = 0          # 乱数シード
        self.timelimit:int  = 1200   # 計算時間制限（秒）
        self.H:list = []             # 各車の訪問可能な位置インデックス

    # データのセット
    def set_data(self,car_num,car_cap,req_num,w_dist,coordinates_dict,seed,form) -> None:
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
        self.H = [str(i) for i in range(2*car_cap+1)]

    # セットされたデータの表示
    def show(self) -> None:
        print('=' * 50)
        print(f'車の台数 : {self.car_num}')
        print(f'車の定員 : {self.car_cap}')
        print(f'リクエスト数 : {self.req_num}')
        print(f'ノード間の距離（生データ） : {self.w_dist}')
        print(f'モデルに使う距離（調整後） : {self.d_uv}')
        print(f'車集合 : {self.D}')
        print(f'リクエスト集合 : {self.R}')
        print(f'全ノード集合 : {self.L}')
        print(f'ノードの訪問順序インデックス集合 : {self.N}')
        print(f'車の初期位置を訪れる順番の集合 : {self.E}')
        print('=' * 50)

    # 数理モデルの構築
    def build_model(self) -> None:
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
            # d0が一番最初に訪問される
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

    # 数理モデルの実行
    def solve(self) -> None:
        solver = pulp.PULP_CBC_CMD(msg=False)
        self.status = self.model.solve(solver)

    def display(self) -> None:
        self.G = nx.DiGraph()
        self.edge_colors = []
        self.node_colors = []

        # 色リスト（車ごとに使う色）
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

        # 車ごとの経路を格納する辞書 (車ノード: [(u,v), (u,v), ...])
        routes = {f'd{i}': [] for i in self.D}

        # まずxuvの値が1のアークを収集し、どの車の経路か判別
        for car in routes.keys():
            current_node = car
            while True:
                # current_nodeから出るアークで変数値が1のものを探す
                next_edge = [ (u,v) for (u,v) in self.xuv.keys()
                               if u == current_node and self.xuv[(u,v)].varValue > 0 and not v.startswith('d')]
                if not next_edge:
                    break
                # 次のノードを取得
                routes[car].append(next_edge[0])
                current_node = next_edge[0][1]
        # グラフにアーク追加＆色設定
        for i, car in enumerate(routes.keys()):
            color = colors[i % len(colors)]
            # 車の初期位置からリストが始まるため、ライトピンクを追加
            self.node_colors.append('lightpink')
            for u, v in routes[car]:
                self.G.add_edge(u, v)
                self.edge_colors.append(color)
                if v[-1].endswith('s'):
                    self.node_colors.append('skyblue')
                if v[-1].endswith('t'):
                    self.node_colors.append('lightgreen')


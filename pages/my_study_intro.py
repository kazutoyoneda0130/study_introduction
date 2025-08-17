import streamlit as st

import my_func as my_func

# 使用できる定式化をセット
form_dict = {
    "定式化1": "$x_{uv},z_{uk},a_{ri}$",
    "定式化2": "$x_{uv},z_{uk}$",
    "定式化3": "$x_{uv},z_{iuk}$",
    "定式化4": "$m_{Pi}$"
}

st.title("ライドシェア運行計画の策定")

st.header("概要")

st.write("""
         ### 導入
         - （日本版）ライドシェアとは
            - タクシー会社によって指示を受けた一般ドライバーが有償で運送サービスを提供するシステム
         - 本研究で取り扱う問題
            - 主体
                - タクシー会社
            - 目的
                - 一台の車が同時に複数リクエストを処理できる状況下において、全ての車の移動時間の合計が最小になるようなルートを作成すること
         ### 問題設定
         - 入力
            - リクエストの集合
                - リクエストはピックアップ場所（お客さんを乗せる場所）とドロップオフ場所（お客さんを下ろす場所）の情報からなる
            - 車の集合
                - 車は初期位置と容量（2 or 3）の情報からなる
         - 出力
            - 各車の訪問順序
         - 条件
            - リクエスト数=車の容量×車の台数
         - 補足
            - 車の容量とは一度の配送で受け入れられるリクエストの数を表す（並行して行うことの出来るリクエストの数ではない）
         """)

st.header("Pythonによる実装")

if st.toggle('問題例作成'):
    st.header("問題例の作成")
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        car_num = st.number_input('車の数',step=1,min_value=1)
    with col2:
        #car_cap = st.number_input('車の容量',step=1,min_value=1)
        car_cap = st.radio("車の容量", [2, 3], horizontal=True)
    # with col3:
    #     req_num = st.number_input('リクエストの数',step=1,min_value=1)
    with col3:
        seed = st.number_input('シード値',step=1,min_value=1)

    req_num = car_num * car_cap

    st.write(f"""
             今回の入力では車の数が{car_num}、容量が{car_cap}であるため、リクエスト数は{req_num}となります。
             
             以下の図では
             - di：車iの初期位置
             - rj_s：リクエストjのピックアップ場所
             - rj_t：リクエストjのドロップオフ場所
             
             を表しています。
             """)

    instance, fig, ax = my_func.create_instance(car_num, car_cap, req_num, seed)
    st.pyplot(fig)

    if st.toggle('ルートの作成'):
        st.header("解の作成")
        st.write("#### 定式化を選択")
        st.write("""
                 定式化を選択すると、作成されたルートとその計算時間を表示します
                 以下に示している定式化は全て厳密解を求めるものなので、得られるルートは全く同じですが、計算時間が異なります
                 - 定式化1,2,3は車の数と容量の積が8以上になると60秒以内に解を求めることが出来ない傾向にあります
                 - 定式化4は車の数と容量の積が30をこえても60秒かかることなく解を求めることが出来る傾向にあります
                 - 結果の図示にも時間がかかるため、結果が表示されるまでの時間と解を求めるのにかかった時間が異なります
                 """)
        form_options = st.radio("",
                                list(form_dict.keys()) +["すべてを実行し、計算時間を比較"],
                                horizontal=True)
        real_time_display, create_char = st.columns([1,2])
        if form_options == "すべてを実行し、計算時間を比較":
            time_dict = {k:0 for k in form_dict.keys()}
            for form in form_dict.keys():
                sol, solve_time = my_func.create_solution(instance, car_num, car_cap, req_num, form=form_dict[form], display_flag=False)
                time_dict[form] = solve_time
                with real_time_display:
                    st.write(f"{form} 実行時間:", solve_time, "秒")
            with create_char:
                fig = my_func.create_bar_chart(time_dict)
                st.plotly_chart(fig)
        else:
            form = form_dict[form_options]
            sol, fig, ax, solve_time = my_func.create_solution(instance, car_num, car_cap, req_num, form, display_flag=True)
            st.pyplot(fig)
            st.write(f"{form_options} 実行時間:", solve_time, "秒")

if st.toggle("備考を表示"):
    st.header("備考")
    st.write("""
             #### 実際の研究で行っていたこと
             - 研究内容
                - 「リクエスト数=車の数×車の容量」という条件が成り立たない問題設定への拡張
                - 各定式化の変数や制約式の数の比較
                - 先行研究で示されていた近似解を求めるアルゴリズムの精度検証
            - プログラム
                - エクセルへの結果書き出し
            ### このページ作成にあたって行ったこと
            - 大学院在学中に使用していたソルバー（数理最適化問題を効率的に解くためのアルゴリズムを実装したソフトウェア）のライセンスが現在使用出来ないため、別のソルバーを使用するためのコード書き直した
            """)
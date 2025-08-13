import streamlit as st
import my_func as my_func
import matplotlib.pyplot as plt
import networkx as nx
import sys
import io

# 使用できる定式化をセット
form_dict = {
    "form1": "$x_{uv},z_{uk}$",
    "form2": "$x_{uv},z_{iuk}$",
    "from3": "$x_{uv},z_{uk},a_{ri}$"}


st.title("研究内容")
st.header("概要")
st.write("""
研究内容の説明をここに記載します。
""")
st.header("問題例の作成")
if st.toggle('問題例作成'):
    # col1,col2,col3,col4 = st.columns([1,1,1,1])
    col1,col2,col4 = st.columns([1,1,1])
    with col1:
        car_num = st.number_input('車の数',step=1,min_value=1)
    with col2:
        car_cap = st.number_input('車のキャパシティー',step=1,min_value=1)
    # with col3:
    #     req_num = st.number_input('リクエストの数',step=1,min_value=1)
    with col4:
        seed = st.number_input('シード値',step=1,min_value=1)

    req_num = car_num * car_cap

    st.write(f"""
             今回のコードではリクエスト数は車の数と車のキャパシティの積とします。
             つまり、今回の入力では車の数が{car_num}、キャパシティが{car_cap}であるため、リクエスト数は{req_num}となります。
             """)

    if car_num * car_cap < req_num:
        st.error("車の数とキャパシティーの積はリクエストの数以上でなければなりません。")
    else:
        instance = my_func.CreateRideShareProblemInstance()
        instance.set_data(car_num=car_num, car_cap=car_cap, req_num=req_num, seed=seed, form="example")
        instance.create(randrange=(0, 100))
        instance.display()
        fig, ax = plt.subplots(figsize=(12,12))
        nx.draw(instance.G, pos=instance.coordinates_dict, with_labels = True, node_size=300, font_size=8, node_color=instance.node_color)
        st.pyplot(fig)

    st.header("解の作成")
    if st.toggle('ルートの作成'):
        form_options = st.radio("定式化の種類を選択",
                                form_dict.keys(),
                                horizontal=True)
        form = form_dict[form_options]
        sol = my_func.MulticapRideshareProblem()
        sol.set_data(car_num,car_cap,req_num,instance.w_dist,instance.coordinates_dict,seed=instance.seed,form=form)
        sol.build_model()
        sol.solve()
        sol.display()
        fig, ax = plt.subplots(figsize=(12,12))
        nx.draw(sol.G, pos=instance.coordinates_dict, with_labels = True, node_size=300, font_size=8, node_color=sol.node_colors, edge_color=sol.edge_colors)
        st.pyplot(fig)
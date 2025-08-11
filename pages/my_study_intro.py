import streamlit as st
import pages.my_func as my_func
import matplotlib.pyplot as plt
import networkx as nx
import sys
import io

st.title("研究内容について")
st.header("概要")
st.write("""
研究内容の説明をここに記載します。
""")
st.header("問題例の作成")
if st.toggle('問題例作成'):
    col1,col2,col3,col4 = st.columns([1,1,1,1])
    with col1:
        car_num = st.number_input('車の数',step=1,min_value=1)
    with col2:
        car_cap = st.number_input('車のキャパシティー',step=1,min_value=1)
    with col3:
        req_num = st.number_input('リクエストの数',step=1,min_value=1)
    with col4:
        seed = st.number_input('シード値',step=1,min_value=1)
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
    st.write('ルートが作成されました！')

import base64
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

import my_func as my_func

# 使用できる定式化をセット
form_dict = {
    "定式化1": "$x_{uv},z_{uk},a_{ri}$",
    "定式化2": "$x_{uv},z_{uk}$",
    "定式化3": "$x_{uv},z_{iuk}$",
    "定式化4": "$m_{Pi}$"
}

st.title("研究内容")
# 研究内容の概要をPDFファイルを埋め込んで表示
st.header("概要")
pdf_path = "修士論文概要.pdf"

with open(pdf_path, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
encoded_pdf = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="800" height="600" type="application/pdf">'
st.markdown(encoded_pdf, unsafe_allow_html=True)

st.header("問題例の作成")
if st.toggle('問題例作成'):
    # col1,col2,col3,col4 = st.columns([1,1,1,1])
    col1,col2,col4 = st.columns([1,1,1])
    with col1:
        car_num = st.number_input('車の数',step=1,min_value=1)
    with col2:
        car_cap = st.number_input('車の容量',step=1,min_value=1)
    # with col3:
    #     req_num = st.number_input('リクエストの数',step=1,min_value=1)
    with col4:
        seed = st.number_input('シード値',step=1,min_value=1)

    req_num = car_num * car_cap

    st.write(f"""
             今回のコードでは「リクエスト数=車の数×車の容量」とします。
             つまり、今回の入力では車の数が{car_num}、容量が{car_cap}であるため、リクエスト数は{req_num}となります。
             """)

    instance, fig, ax = my_func.create_instance(car_num, car_cap, req_num, seed)
    st.pyplot(fig)

    st.header("解の作成")
    if st.toggle('ルートの作成'):
        st.write("#### 定式化を選択してください。")
        form_options = st.radio("",
                                list(form_dict.keys()) +["すべてを実行し、計算時間を比較"],
                                horizontal=True)
        if form_options == "すべてを実行し、計算時間を比較":
            for form in reversed(list(form_dict.keys())):
                if form == "定式化4":
                    # 結果の図示
                    sol, fig, ax, solve_time = my_func.create_solution(instance, car_num, car_cap, req_num, form=form_dict[form], display_flag=True)
                    st.pyplot(fig)
                else:
                    sol, solve_time = my_func.create_solution(instance, car_num, car_cap, req_num, form=form_dict[form], display_flag=False)
                st.write(f"{form} 実行時間:", solve_time, "秒")
        else:
            form = form_dict[form_options]
            sol, fig, ax, solve_time = my_func.create_solution(instance, car_num, car_cap, req_num, form, display_flag=True)
            st.pyplot(fig)
            st.write("実行時間:", solve_time, "秒")
import streamlit as st
import base64

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

if st.toggle('問題例作成'):
    st.header("問題例の作成")
    # col1,col2,col3,col4 = st.columns([1,1,1,1])
    col1,col2,col4 = st.columns([1,1,1])
    with col1:
        car_num = st.number_input('車の数',step=1,min_value=1)
    with col2:
        #car_cap = st.number_input('車の容量',step=1,min_value=1)
        car_cap = st.radio("車の容量", [2, 3], horizontal=True)
    # with col3:
    #     req_num = st.number_input('リクエストの数',step=1,min_value=1)
    with col4:
        seed = st.number_input('シード値',step=1,min_value=1)

    req_num = car_num * car_cap

    st.write(f"""
             「リクエスト数=車の数×車の容量」とします。
             つまり、今回の入力では車の数が{car_num}、容量が{car_cap}であるため、リクエスト数は{req_num}となります。
             """)

    instance, fig, ax = my_func.create_instance(car_num, car_cap, req_num, seed)
    st.pyplot(fig)

    if st.toggle('ルートの作成'):
        st.header("解の作成")
        st.write("#### 定式化を選択")
        st.write("""
                 定式化を選択すると、作成されたルートとその計算時間を表示します。
                 以下に示している定式化は全て厳密解を求めるものなので、得られるルートは全く同じですが、計算時間が異なります。
                 - 定式化1,2,3は車の数と容量の積が6以上になると計算時間が増大する傾向にあります。
                 - 定式化4は車の数と容量の積が20程度でも計算時間が増大することなく解を求めることができます。
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
            st.write("実行時間:", solve_time, "秒")

if st.toggle("備考を表示"):
    st.header("備考")
    st.write("""
             #### 実際の研究で行っていたこと
             - 研究内容
                - 「リクエスト数=車の数×車の容量」という条件が成り立たない問題設定への拡張
                - 各定式化の変数や制約式の数の比較
                - 先行研究で示されていた近似解を求めるアルゴリズムの精度検証
            - プログラム
                - 大学院在学中に使用していたソルバーのライセンスが切れてしまっているため、別のソルバーを使用するためにコードを一部書き直した
                - エクセルへの結果書き出し
            """)
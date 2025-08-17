import streamlit as st
import sys
import io

st.title("数理最適化について")

st.header("数理最適化とは")
st.write("""
数理最適化とは一言でいうと、「数学を使って一番いい答えを見つける技術」です。
条件（制約条件）があって、その中で目的（利益最大化・コスト最小化など）を達成したい時に使用出来ます。
例えば、
- スーパーで食材を買う時に、予算内で（条件）一番多くの食材を買いたい（目的）
- 旅行計画で、移動時間を最小限に抑えつつ（目的）訪れたい観光地を全て回りたい（条件）

など、日常生活の様々な場面で数理最適化は利用出来ます。また、ビジネスの現場でも在庫管理・物流の最適化・スケジューリングなど幅広い分野で活用されています。
""")

st.header("数理最適化とPythonの関係について")
st.write("""
Pythonには数理最適化を解くためのライブラリが沢山提供されています。
例えば、`PuLP`、`SciPy.optimize`、`OR-Tools`などがあります。
これらを使うことで、最適化問題をPythonで記述し、解くことができます。
""")

st.header("数理最適化問題の例とその問題のPythonでの実行例")
st.write("""
ある会社は商品Iと商品Jを生産しています。以下の条件を守りながら利益を最大化するためには、生産量をどのように決めれば良いでしょうか？
- 商品Iの利益は1個あたり3万円、商品Jの利益は1個あたり2万円
- 商品Iを1個生産するのに資源Aが2kg、資源Bが1kg必要
- 商品Jを1個生産するのに資源Aが1kg、資源Bが1kg必要
- 資源Aは合計で100kgまで、資源Bは合計で80kgまで使用可能
""")
st.write("### 数式での記述")
st.write("xを商品Iの生産量、yを商品Jの生産量とすると、以下のように表現できます。")
st.write("##### 目的関数")
st.latex(r"\max \quad 3x + 2y")

st.write("##### 制約条件")
st.latex(r"2x + y \leq 100")
st.latex(r"x + y \leq 80")

st.write("##### コード例とその結果")

code = '''
import pulp

# 問題の定義
# 最大化問題であることを定義
prob = pulp.LpProblem("Simple LP Problem", pulp.LpMaximize)

# 変数の定義
# xとyはそれぞれ商品Iと商品Jの生産量
# xとyは0以上の整数であることを定義
x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)

# 目的関数
# 利益の最大化
prob += 3 * x + 2 * y

# 制約条件
# 資源Aと資源Bの制約
prob += 2 * x + y <= 100
prob += x + y <= 80

# 解く
prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print("x =", pulp.value(x))
print("y =", pulp.value(y))
print("Objective =", pulp.value(prob.objective))
print(f"最適な生産量は「商品Iが{int(pulp.value(x))}個、商品Jが{int(pulp.value(y))}個」で、その時の利益は {int(pulp.value(prob.objective))}万円です。")
'''

st.code(code, language='python')

if st.button("コードを実行する"):
    # 標準出力をキャプチャ
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(code, globals())
        output = new_stdout.getvalue()
    except Exception as e:
        output = f"エラーが発生しました:\n{e}"
    finally:
        sys.stdout = old_stdout

    st.text_area("実行結果", output, height=200)

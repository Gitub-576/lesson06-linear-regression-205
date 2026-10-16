
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고 "
    "회귀 직선을 이용해 선택한 연도의 예상 기온을 확인합니다."
)

# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(missing)}"
        )

    # 날짜와 기온을 숫자 형태로 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일 수와 연평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 하려면 유효한 연도가 2개 이상 필요합니다.")
    st.stop()

# -----------------------------
# 회귀 분석 함수
# -----------------------------
def fit_regression(data):
    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]

    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
        "count": len(data),
        "start": int(data["연도"].min()),
        "end": int(data["연도"].max())
    }


# 전체 기간 회귀 분석
overall = fit_regression(yearly)

# 최근 20년: 유효한 데이터 중 가장 최근 연도부터 20년 범위
latest_year = int(yearly["연도"].max())
recent_start = latest_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start)
    & (yearly["연도"] <= latest_year)
].copy()

if len(recent) >= 2:
    recent_model = fit_regression(recent)
else:
    recent_model = None

# -----------------------------
# 주요 지표
# -----------------------------
st.subheader("📈 전체 기간 분석")

c1, c2, c3 = st.columns(3)

c1.metric("분석에 사용한 연도", f"{overall['count']}년")
c2.metric(
    "분석 시작 연도",
    f"{overall['start']}년"
)
c3.metric(
    "분석 끝 연도",
    f"{overall['end']}년"
)

st.caption(
    "2025년까지의 자료 중 관측일이 300일 이상인 연도만 사용합니다."
)

# -----------------------------
# 기울기 비교: 100년당 기온 변화
# -----------------------------
st.subheader("🌡️ 100년당 기온 변화 비교")

left, right = st.columns(2)

with left:
    st.markdown("#### 전체 기간")
    st.metric(
        "100년에 기온 변화",
        f"{overall['slope'] * 100:+.2f} °C"
    )
    st.caption(
        f"{overall['start']}~{overall['end']}년 자료 "
        f"({overall['count']}년)"
    )

with right:
    st.markdown("#### 최근 20년")
    if recent_model is not None:
        st.metric(
            "100년에 기온 변화",
            f"{recent_model['slope'] * 100:+.2f} °C"
        )
        st.caption(
            f"{recent_model['start']}~{recent_model['end']}년 자료 "
            f"({recent_model['count']}년)"
        )
    else:
        st.warning("최근 기간에 유효한 연도가 부족합니다.")

st.caption(
    "기울기에 100을 곱해 100년당 변화량으로 환산했습니다. "
    "양수는 상승, 음수는 하락을 뜻합니다. "
    "최근 20년 분석은 가장 최근 유효 연도부터 20년 범위를 사용합니다."
)

if recent_model is not None:
    difference = (
        recent_model["slope"] - overall["slope"]
    ) * 100

    if difference > 0:
        st.write(
            f"최근 기간의 기울기가 전체 기간보다 "
            f"100년당 {difference:.2f}°C 더 높습니다."
        )
    elif difference < 0:
        st.write(
            f"최근 기간의 기울기가 전체 기간보다 "
            f"100년당 {abs(difference):.2f}°C 낮습니다."
        )
    else:
        st.write("두 기간의 기울기는 같습니다.")

# -----------------------------
# 산점도 + 전체 회귀 직선
# -----------------------------
st.subheader("📊 연도별 평균기온과 회귀 직선")

line_x = np.linspace(
    yearly["연도"].min(),
    yearly["연도"].max(),
    300
)
line_y = overall["slope"] * line_x + overall["intercept"]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="연평균기온",
    marker=dict(size=6),
    customdata=yearly[["관측일수"]].to_numpy(),
    hovertemplate=(
        "연도: %{x}년<br>"
        "연평균기온: %{y:.2f}°C<br>"
        "관측일수: %{customdata[0]}일"
        "<extra></extra>"
    )
))

fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="전체 기간 회귀 직선",
    line=dict(width=3),
    hovertemplate=(
        "연도: %{x:.1f}년<br>"
        "추세 기온: %{y:.2f}°C"
        "<extra></extra>"
    )
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례"
)

st.plotly_chart(fig, width="stretch")

st.write(
    f"**상관계수:** {overall['correlation']:.4f}"
)

st.caption(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "양수이면 대체로 시간이 지날수록 기온이 높아지는 관계입니다."
)

# -----------------------------
# 연도 선택 및 예상 기온
# -----------------------------
st.subheader("🔮 연도별 예상 연평균기온")

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=min(max(latest_year, 1900), 2100),
    step=1
)

predicted_temp = (
    overall["slope"] * selected_year
    + overall["intercept"]
)

st.markdown(
    f"<div style='text-align:center; padding:20px;'>"
    f"<div style='font-size:22px;'>"
    f"{selected_year}년 예상 연평균기온"
    f"</div>"
    f"<div style='font-size:56px; font-weight:bold;'>"
    f"{predicted_temp:.2f} °C"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

st.caption(
    "예상 기온은 전체 기간의 회귀 직선을 선택한 연도까지 연장해 계산한 값입니다. "
    "특히 관측 범위를 크게 벗어나는 미래 연도의 결과는 불확실성이 큽니다."
)

# -----------------------------
# 선택 연도의 예측 위치 표시
# -----------------------------
prediction_fig = go.Figure()

prediction_fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    marker=dict(size=5),
    hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>"
))

prediction_x = np.linspace(
    min(float(yearly["연도"].min()), 1900),
    2100,
    400
)
prediction_y = (
    overall["slope"] * prediction_x + overall["intercept"]
)

prediction_fig.add_trace(go.Scatter(
    x=prediction_x,
    y=prediction_y,
    mode="lines",
    name="회귀 직선 연장",
    line=dict(width=3),
    hovertemplate="%{x:.0f}년<br>%{y:.2f}°C<extra></extra>"
))

prediction_fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name="선택한 연도의 예상 기온",
    marker=dict(size=14, symbol="star"),
    hovertemplate=(
        f"{selected_year}년 예상 기온<br>"
        "%{y:.2f}°C<extra></extra>"
    )
))

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    hovermode="closest"
)

st.plotly_chart(prediction_fig, width="stretch")

# -----------------------------
# 데이터 확인
# -----------------------------
with st.expander("📋 분석에 사용된 연도별 데이터 보기"):
    display_data = yearly.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    st.dataframe(display_data, width="stretch")

st.caption(
    "자료 출처: 기상청 서울 기온 CSV 데이터. "
    "회귀 직선은 과거의 선형 추세를 나타내며 미래의 실제 기온을 보장하지 않습니다."
)

import queue
import time
import io, av, time, math, base64
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Photometry Lab", layout="wide")

# ------------- Yardımcılar -------------
def linear_fit_r2(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.zeros_like(x, dtype=bool)
    sx, sy = np.sum(x), np.sum(y)
    sxx, sxy = np.sum(x*x), np.sum(x*y)
    den = (n*sxx - sx*sx)
    if den == 0:
        return np.nan, np.nan, np.nan, np.zeros_like(x, dtype=bool)
    m = (n*sxy - sx*sy) / den
    b = (sy - m*sx) / n
    y_hat = m*x + b
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_hat)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2, np.ones_like(x, dtype=bool)

def to_absorbance(I: np.ndarray, I0: float):
    I = np.clip(I, 1e-6, None)
    if not np.isfinite(I0) or I0 <= 0:
        return np.full_like(I, np.nan, dtype=float)
    return np.log10(I0 / I)  # A = log10(I0/I)

def df_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">CSV indir</a>'
    return href

# ------------- UI: Sidebar -------------
st.sidebar.title("🧪 Photometry Lab")
st.sidebar.markdown("**Mod:**")
mode = st.sidebar.radio(
    label="Analiz modu",
    options=["Intensity", "Absorbance — Kinetics", "Absorbance — Endpoint"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Giriş:** Kamera veya Video Yükle")

fps = st.sidebar.slider("Örnekleme (fps)", 1, 10, 2)
duration_limit = st.sidebar.slider("Kayıt süresi sınırı (dk)", 1, 60, 5)

# ------------- Sekmeler -------------
tab_live, tab_upload, tab_analyze = st.tabs(["🔴 Canlı Ölçüm", "📂 Video Analizi", "📊 Analiz/Grafik"])

# ------------- 1) CANLI ÖLÇÜM -------------
with tab_live:
    st.markdown("**Kameradan canlı ölçüm** — ortalama R,G,B ve Gray zaman serisi üretir.")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    st.session_state.setdefault("live_t", [])
    st.session_state.setdefault("live_R", [])
    st.session_state.setdefault("live_G", [])
    st.session_state.setdefault("live_B", [])
    start_ts = st.session_state.get("start_ts", None)

    class VideoProcessor:
        def __init__(self):
            self.frame_count = 0

        def recv(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")  # HxWx3 BGR
            h, w, _ = img.shape
            # Ortalama RGB
            b = img[:, :, 0].mean()
            g = img[:, :, 1].mean()
            r = img[:, :, 2].mean()
            # Zaman damgası
            global start_ts
            if start_ts is None:
                start_ts = time.time()
                st.session_state["start_ts"] = start_ts
            t = time.time() - start_ts

            # Örnekleme hızı kontrolü:
            # stream 30 fps olsa da, sadece (fps) hızında örnek alalım
            self.frame_count += 1
            # Basit subsample: her n karede bir kayıt
            nskip = max(1, round(30 / fps))
            if self.frame_count % nskip == 0:
                st.session_state["live_t"].append(t)
                st.session_state["live_R"].append(r)
                st.session_state["live_G"].append(g)
                st.session_state["live_B"].append(b)

            return frame

    colL, colR = st.columns([2, 1])
    with colL:
        ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
        )
    with colR:
        if st.button("⏹️ Sıfırla / Yeni Kayıt"):
            st.session_state["live_t"] = []
            st.session_state["live_R"] = []
            st.session_state["live_G"] = []
            st.session_state["live_B"] = []
            st.session_state["start_ts"] = None
            st.success("Temizlendi.")

        if len(st.session_state["live_t"]) > 3:
            df_live = pd.DataFrame({
                "time": st.session_state["live_t"],
                "R": st.session_state["live_R"],
                "G": st.session_state["live_G"],
                "B": st.session_state["live_B"],
            })
            df_live["Gray"] = 0.114*df_live["B"] + 0.587*df_live["G"] + 0.299*df_live["R"]
            st.markdown(df_download_link(df_live, "live_rgb.csv"), unsafe_allow_html=True)

    # Canlı grafik
    if len(st.session_state["live_t"]) > 1:
        dfp = pd.DataFrame({
            "time": st.session_state["live_t"],
            "R": st.session_state["live_R"],
            "G": st.session_state["live_G"],
            "B": st.session_state["live_B"],
        })
        dfp["Gray"] = 0.114*dfp["B"] + 0.587*dfp["G"] + 0.299*dfp["R"]

        fig = go.Figure()
        for name, color in [("R", "red"), ("G", "green"), ("B", "blue"), ("Gray", "black")]:
            fig.add_trace(go.Scatter(x=dfp["time"], y=dfp[name], mode="lines", name=name, line=dict(color=color)))
        fig.update_layout(height=350, xaxis_title="Zaman (s)", yaxis_title="Mean Intensity (0-255)", template="simple_white")
        st.plotly_chart(fig, use_container_width=True)

# ------------- 2) ÖNCEKİ VİDEODAN ANALİZ -------------
with tab_upload:
    st.markdown("**MP4/AVI video yükle** → ortalama R,G,B ve Gray çıkar.")
    f = st.file_uploader("Video seç", type=["mp4", "mov", "avi", "mkv"])
    if f is not None:
        # OpenCV ile kare okuma
        import cv2
        bytes_data = f.read()
        tmp = f"temp_{int(time.time())}.mp4"
        with open(tmp, "wb") as wf: wf.write(bytes_data)
        cap = cv2.VideoCapture(tmp)
        times, R, G, B = [], [], [], []
        fps_read = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # subsample: hedef fps
            if frame_idx % max(1, round(fps_read / fps)) == 0:
                b = frame[:, :, 0].mean()
                g = frame[:, :, 1].mean()
                r = frame[:, :, 2].mean()
                times.append((frame_idx / fps_read))
                R.append(r); G.append(g); B.append(b)
            frame_idx += 1
        cap.release()
        df = pd.DataFrame({"time": times, "R": R, "G": G, "B": B})
        df["Gray"] = 0.114*df["B"] + 0.587*df["G"] + 0.299*df["R"]
        st.success(f"Kare sayısı: {frame_idx} | örneklenen: {len(df)}")
        st.dataframe(df.head())
        st.session_state["uploaded_df"] = df

# ------------- 3) ANALİZ/GRAFİK -------------
with tab_analyze:
    st.markdown("**Grafik + Blank + Fit / Endpoint**")
    source = st.radio("Veri kaynağı", ["Canlı (üst sekme)", "Yüklenen video"], index=0, horizontal=True)
    if source == "Canlı (üst sekme)":
        if len(st.session_state["live_t"]) < 3:
            st.info("Canlı ölçümden yeterli veri yok. Üst sekmede kayıt başlat.")
            st.stop()
        df = pd.DataFrame({
            "time": st.session_state["live_t"],
            "R": st.session_state["live_R"],
            "G": st.session_state["live_G"],
            "B": st.session_state["live_B"],
        })
        df["Gray"] = 0.114*df["B"] + 0.587*df["G"] + 0.299*df["R"]
    else:
        df = st.session_state.get("uploaded_df")
        if df is None:
            st.info("Önce video yükle.")
            st.stop()

    # Blank seçimi
    st.subheader("Blank (I0) seçimi")
    tmin, tmax = float(df["time"].min()), float(df["time"].max())
    blank_range = st.slider("Blank aralığı (I0 için)", tmin, tmax, (tmin, min(tmin + (tmax - tmin)*0.1, tmax)))
    df_blank = df[(df["time"] >= blank_range[0]) & (df["time"] <= blank_range[1])]
    I0 = {ch: float(df_blank[ch].mean()) for ch in ["R", "G", "B", "Gray"]}
    st.caption(f"I0 (mean): R={I0['R']:.2f}, G={I0['G']:.2f}, B={I0['B']:.2f}, Gray={I0['Gray']:.2f}")

    # Analiz aralığı
    st.subheader("Analiz aralığı")
    default_a = blank_range[1]
    analyze_range = st.slider("Fit/Endpoint aralığı", tmin, tmax, (default_a, min(default_a + (tmax-tmin)*0.3, tmax)))
    a, b = analyze_range

    # Mod dönüşümü
    work = df.copy()
    ylabel = "Mean Intensity (0-255)"
    if mode != "Intensity":
        ylabel = "Absorbance (AU)"
        for ch in ["R", "G", "B", "Gray"]:
            work[ch] = to_absorbance(work[ch].to_numpy(), I0[ch])

    # Grafik
    fig = go.Figure()
    for ch, color in [("R","red"), ("G","green"), ("B","blue"), ("Gray","black")]:
        fig.add_trace(go.Scatter(x=work["time"], y=work[ch], mode="lines", name=ch, line=dict(color=color)))
    fig.add_vrect(x0=blank_range[0], x1=blank_range[1], fillcolor="orange", opacity=0.15, line_width=0, annotation_text="BLANK", annotation_position="top left")
    fig.add_vrect(x0=a, x1=b, fillcolor="deepskyblue", opacity=0.12, line_width=0, annotation_text="ANALYSIS", annotation_position="top left")
    fig.update_layout(height=420, xaxis_title="Zaman (s)", yaxis_title=ylabel, template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

    # Sonuçlar
    st.subheader("Sonuçlar")
    rows = []
    out_fit = { "time": work["time"].to_numpy() }
    for ch in ["R","G","B","Gray"]:
        sub = work[(work["time"]>=a) & (work["time"]<=b)]
        x = sub["time"].to_numpy()
        y = sub[ch].to_numpy()

        if mode == "Absorbance — Endpoint":
            mean_val = float(np.nanmean(y)) if len(y)>0 else np.nan
            rows.append([ch, mode, np.nan, np.nan, np.nan, a, b, mean_val])
            yfit = np.full_like(work["time"].to_numpy(), np.nan, dtype=float)
            mask = (work["time"]>=a) & (work["time"]<=b)
            yfit[mask] = mean_val
            out_fit[f"{ch}_fit"] = yfit
        else:
            m, c, r2, okmask = linear_fit_r2(x, y)
            rows.append([ch, mode, m, c, r2, a, b, np.nan])
            yfit = np.full_like(work["time"].to_numpy(), np.nan, dtype=float)
            mask = (work["time"]>=a) & (work["time"]<=b)
            yfit[mask] = (m*work["time"].to_numpy()[mask] + c) if np.isfinite(m) else np.nan
            out_fit[f"{ch}_fit"] = yfit

    summary = pd.DataFrame(rows, columns=["Series","Analysis_Type","Slope_m","Intercept_b","R2","t1","t2","Mean_Value"])
    merged = pd.DataFrame({"time": df["time"]})
    for ch in ["R","G","B","Gray"]:
        merged[f"{ch}_int"] = df[ch]
    if mode != "Intensity":
        for ch in ["R","G","B","Gray"]:
            merged[f"{ch}_abs"] = to_absorbance(df[ch].to_numpy(), I0[ch])
    else:
        for ch in ["R","G","B","Gray"]:
            merged[f"{ch}_abs"] = np.nan
    for k,v in out_fit.items():
        merged[k] = v

    st.dataframe(summary, use_container_width=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Fit Özeti (CSV)", summary.to_csv(index=False).encode("utf-8"), file_name="fit_summary.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Birleşik Veri (CSV)", merged.to_csv(index=False).encode("utf-8"), file_name="merged_data.csv", mime="text/csv")
    with c3:
        # Grafiği PNG olarak indirme (basit ekran görüntüsü yaklaşımı)
        png_bytes = fig.to_image(format="png", width=1200, height=500, scale=2)
        st.download_button("🖼️ Grafiği PNG indir", png_bytes, file_name="analysis.png", mime="image/png")


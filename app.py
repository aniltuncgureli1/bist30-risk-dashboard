import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date
import traceback

# ==========================================
# SAYFA YAPILANDIRMASI VE CSS
# ==========================================
st.set_page_config(page_title="BIST 30 Risk Portalı", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 35px !important; font-weight: bold !important; color: #00d4b2 !important; }
div[data-testid="stMetricLabel"] { font-size: 18px !important; font-weight: bold !important; color: #f0f2f6 !important; }
h1, h2, h3 { color: #f0f2f6 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 20px; }
</style>
""", unsafe_allow_html=True)

TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS",
    "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS",
    "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "MGROS.IS",
    "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
    "TKFEN.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
]

# ==========================================
# VERİ ÇEKME VE RİSK HESAPLAMA MOTORU
# ==========================================
@st.cache_data(ttl=3600, show_spinner="Yahoo Finance verileri çekiliyor ve metrikler hesaplanıyor...")
def verileri_hazirla():
    try:
        raw = yf.download(TICKERS, period="10y", auto_adjust=True, progress=False)
        df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        results = []
        for ticker in TICKERS:
            col = ticker if ticker in df.columns else ticker.replace(".IS", "")
            if col not in df.columns:
                continue
            prices = df[col].dropna()
            if len(prices) < 252:
                continue

            log_ret = np.log(prices / prices.shift(1)).dropna()
            volatility = float(log_ret.std() * np.sqrt(252))

            cum = (1 + log_ret).cumprod()
            drawdown = (cum - cum.cummax()) / cum.cummax()
            max_dd = float(abs(drawdown.min()))

            last_252 = prices.iloc[-252:]
            
            results.append({
                "ticker": ticker.replace(".IS", ""),
                "full_ticker": ticker,
                "volatility": volatility,
                "max_drawdown": max_dd,
                "current_price": float(prices.iloc[-1]),
                "high_52w": float(last_252.max()),
                "low_52w": float(last_252.min()),
            })

        if not results:
            return None, None

        vol = np.array([r["volatility"] for r in results])
        dd = np.array([r["max_drawdown"] for r in results])
        vol_n = (vol - vol.min()) / (vol.max() - vol.min() + 1e-10)
        dd_n = (dd - dd.min()) / (dd.max() - dd.min() + 1e-10)
        scores = 0.6 * vol_n + 0.4 * dd_n

        q25, q50, q75 = np.percentile(scores, [25, 50, 75])

        for i, r in enumerate(results):
            s = float(scores[i])
            r["risk_score"] = s
            if s <= q25:
                r["category"], r["category_sub"], r["category_level"], r["color"] = "Çok Güvenilir", "Defansif", 1, "#00d4b2"
            elif s <= q50:
                r["category"], r["category_sub"], r["category_level"], r["color"] = "Güvenilir", "Dengeli", 2, "#29b5e8"
            elif s <= q75:
                r["category"], r["category_sub"], r["category_level"], r["color"] = "Az Güvenilir", "Dinamik", 3, "#ffa421"
            else:
                r["category"], r["category_sub"], r["category_level"], r["color"] = "Güvenilmez", "Agresif", 4, "#ff4b4b"

        results_df = pd.DataFrame(results).sort_values(by="risk_score")
        clean_df = df.rename(columns=lambda c: c.replace(".IS", "") if ".IS" in str(c) else c)
        
        return results_df, clean_df
    
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return None, None

risk_df, price_df = verileri_hazirla()

# ==========================================
# MAKİNE ÖĞRENMESİ (RIDGE) TAHMİN MOTORU
# ==========================================
@st.cache_data(show_spinner=False)
def hisse_tahmini_yap(ticker):
    prices = price_df[ticker].dropna()
    df_f = pd.DataFrame({"price": prices})
    
    for lag in [1, 2, 3, 5, 10, 20]:
        df_f[f"lag_{lag}"] = df_f["price"].shift(lag)
    df_f["ma_20"] = df_f["price"].rolling(20).mean()
    df_f["ma_50"] = df_f["price"].rolling(50).mean()
    df_f["ma_200"] = df_f["price"].rolling(200).mean()
    df_f = df_f.dropna()
    
    X = df_f.drop("price", axis=1).values
    y = df_f["price"].values
    split = max(len(X) - 126, int(len(X) * 0.8))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_te = scaler.transform(X[split:])
    
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y[:split])
    backtest = model.predict(X_te)
    
    # Hata Metrikleri
    mape = np.mean(np.abs((y[split:] - backtest) / y[split:])) * 100
    basari_orani = max(0, 100 - mape)

    today = date.today()
    year_end = date(2026, 12, 31)
    trading_days = max(1, int((year_end - today).days * 252 / 365))

    price_buf = list(df_f["price"].iloc[-200:].values)
    forecast_prices = []
    
    for _ in range(trading_days):
        lags = [price_buf[-i] for i in [1, 2, 3, 5, 10, 20]]
        ma20 = float(np.mean(price_buf[-20:]))
        ma50 = float(np.mean(price_buf[-50:])) if len(price_buf) >= 50 else float(np.mean(price_buf))
        ma200 = float(np.mean(price_buf[-200:])) if len(price_buf) >= 200 else float(np.mean(price_buf))
        
        feat = np.array(lags + [ma20, ma50, ma200]).reshape(1, -1)
        pred = float(model.predict(scaler.transform(feat))[0])
        forecast_prices.append(pred)
        price_buf.append(pred)

    hist_dates = df_f.index
    backtest_dates = df_f.index[split:]
    forecast_dates = pd.bdate_range(pd.Timestamp.today(), periods=trading_days)
    
    return hist_dates, df_f["price"].values, backtest_dates, backtest, forecast_dates, forecast_prices, basari_orani

# ==========================================
# KULLANICI ARAYÜZÜ (UI) BAŞLANGICI
# ==========================================
if risk_df is None or price_df is None:
    st.stop()

st.title("📊 Bitirme Projesi: BIST 30 Risk ve Volatilite Ağı")

tab1, tab2 = st.tabs(["🔍 Bireysel Hisse Analizi & Tahmin", "💼 Portföy Risk Optimizasyonu"])

# --- TAB 1: BİREYSEL HİSSE ---
with tab1:
    hisse_listesi = risk_df["ticker"].tolist()
    
    col_sel, _ = st.columns([1, 3])
    selected_ticker = col_sel.selectbox("İncelenecek Hisse:", hisse_listesi)
    
    hisse_verisi = risk_df[risk_df["ticker"] == selected_ticker].iloc[0]
    
    st.markdown("<hr style='border: 1px solid #444; margin: 15px 0;'>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Güncel Fiyat", f"{hisse_verisi['current_price']:.2f} TL")
    c2.metric("Yıllık Volatilite", f"%{hisse_verisi['volatility']*100:.2f}")
    c3.metric("Maksimum Düşüş", f"%{hisse_verisi['max_drawdown']*100:.2f}")
    
    html_card = f"""
    <div style="background-color: #2b2b2b; padding: 15px; border-radius: 12px; border-left: 6px solid {hisse_verisi['color']}; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <h5 style="color: white; margin: 0; font-size: 20px;">Risk Sınıfı: <span style="color: {hisse_verisi['color']}">{hisse_verisi['category']} ({hisse_verisi['category_sub']})</span></h5>
    </div>
    """
    c4.markdown(html_card, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"🔮 2026 Yıl Sonu Tahmini (Ridge Regresyon)", anchor=False)
    
    h_dates, h_prices, b_dates, b_prices, f_dates, f_prices, basari = hisse_tahmini_yap(selected_ticker)
    
    beklenen_getiri = ((f_prices[-1] - hisse_verisi['current_price']) / hisse_verisi['current_price']) * 100
    
    tc1, tc2 = st.columns(2)
    tc1.metric("Modelin Yıl Sonu Hedefi", f"{f_prices[-1]:.2f} TL", delta=f"%{beklenen_getiri:.2f} Beklenen Getiri")
    tc2.metric("🎯 Model Doğruluk Oranı", f"%{basari:.1f}", help="Son 6 aylık veriler üzerinde yapılan geriye dönük doğrulama (backtesting) başarısı.")
    
    # Grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_dates, y=h_prices, mode='lines', name='Gerçekleşen Fiyat', line=dict(color='#00d4b2', width=2)))
    fig.add_trace(go.Scatter(x=b_dates, y=b_prices, mode='lines', name='Backtest (Test Seti)', line=dict(color='#ffa421', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=f_dates, y=f_prices, mode='lines', name='2026 Projeksiyonu', line=dict(color='#ff4b4b', width=4)))
    
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PORTFÖY HESAPLAYICI ---
with tab2:
    st.subheader("Portföy Kompozisyonu", anchor=False)
    st.write("FastAPI tabanındaki algoritmanın dinamik portföy ağırlıklandırma test alanı.")
    
    p_col1, p_col2 = st.columns([2, 1])
    
    with p_col1:
        secilen_portfoy = st.multiselect("Portföye Hisse Ekle (En az 2):", hisse_listesi, default=["AKBNK", "THYAO"])
        
        if len(secilen_portfoy) >= 2:
            agirliklar = []
            cols = st.columns(len(secilen_portfoy))
            for idx, hisse in enumerate(secilen_portfoy):
                w = cols[idx].number_input(f"{hisse} Ağırlığı", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                agirliklar.append(w)
                
            w_array = np.array(agirliklar)
            w_array = w_array / w_array.sum()
            
            # Matris Hesaplamaları
            prices = price_df[secilen_portfoy].ffill().dropna(how="all")
            returns = prices.pct_change().dropna()
            
            cov = returns.cov() * 252
            cov_vals = np.nan_to_num(cov.values, nan=0.0)
            port_var = float(w_array @ cov_vals @ w_array)
            port_vol = float(np.sqrt(max(port_var, 0)))
            
            port_ret = (returns * w_array).sum(axis=1)
            cum = (1 + port_ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = float(abs(dd.min()))
            
            w_score = 0.0
            for i, ticker in enumerate(secilen_portfoy):
                stock_data = risk_df[risk_df["ticker"] == ticker].iloc[0]
                w_score += float(w_array[i]) * stock_data["risk_score"]
                
            if w_score <= 0.25: p_cat, p_color = "Çok Güvenilir (Defansif)", "#00d4b2"
            elif w_score <= 0.50: p_cat, p_color = "Güvenilir (Dengeli)", "#29b5e8"
            elif w_score <= 0.75: p_cat, p_color = "Az Güvenilir (Dinamik)", "#ffa421"
            else: p_cat, p_color = "Güvenilmez (Agresif)", "#ff4b4b"
            
            st.markdown("<hr>", unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Kümülatif Portföy Volatilitesi", f"%{port_vol*100:.2f}")
            pc2.metric("Portföy Maksimum Düşüşü", f"%{max_dd*100:.2f}")
            pc3.markdown(f"**Ağırlıklı Risk Skalası:**<br><span style='color:{p_color}; font-size:24px; font-weight:bold;'>{p_cat}</span>", unsafe_allow_html=True)
            
            # Portföy Dağılımı Grafiği
            fig_pie = px.pie(values=w_array*100, names=secilen_portfoy, title="Ağırlık Dağılımı", hole=0.4)
            fig_pie.update_layout(height=400, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.warning("Hesaplama yapabilmek için en az 2 hisse seçmelisiniz.")

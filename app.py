import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from veri_isleme import risk_metriklerini_hesapla

st.set_page_config(
    page_title="BIST 30 Risk Portalı", 
    layout="wide"
)

# ==========================================
# CSS İLE GÖRSEL DÜZENLEMELER
# ==========================================
st.markdown("""
<style>
/* Font ve Genel Layout */
html, body, p, li, span, .stMarkdown { 
    font-size: 26px !important; 
    line-height: 1.6 !important; 
}

/* Metrik Değerleri */
div[data-testid="stMetricValue"] { 
    font-size: 50px !important; 
    font-weight: 900 !important; 
    color: #00d4b2 !important; 
    padding-bottom: 10px !important; 
}

/* Metrik Başlıkları */
div[data-testid="stMetricLabel"] { 
    font-size: 24px !important; 
    font-weight: bold !important; 
    color: #f0f2f6 !important; 
}

/* Ana ve Alt Başlıklar */
h1 { 
    font-size: 50px !important; 
    color: #ff4b4b !important; 
    margin-bottom: 10px !important; 
}
h2 { 
    font-size: 40px !important; 
    margin-top: 15px !important; 
    margin-bottom: 10px !important; 
}
h3 { 
    font-size: 32px !important; 
    margin-top: 15px !important; 
    margin-bottom: 10px !important; 
}

/* Sol Menü Ayarları */
div[data-testid="stSidebar"] label { 
    font-size: 22px !important; 
    line-height: 1.5 !important; 
}

/* Grafik Kontrolleri ve Büyütme */
.modebar-container { 
    transform: scale(1.8) !important; 
    transform-origin: top right !important; 
}

/* İstenmeyen Link (Anchor) İkonlarının Gizlenmesi */
a.header-anchor { 
    display: none !important; 
}
h1 a, h2 a, h3 a { 
    display: none !important; 
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ANA UYGULAMA BAŞLIĞI VE AÇIKLAMASI
# ==========================================
st.title("📊 BIST 30 Risk ve Volatilite Analiz Portalı", anchor=False)
st.write(
    "Bu portal, BIST 30 hisselerinin son 10 yıllık fiyat hareketlerini inceleyerek risk skorları üretir, "
    "çeyreklik eşiklere göre sınıflandırır ve Ridge Regresyon modeliyle 2026 yıl sonu tahmini yapar."
)

@st.cache_data
def verileri_yukle():
    sonuclar, fiyatlar = risk_metriklerini_hesapla("bist30_10yillik_fiyatlar.xlsx")
    return sonuclar, fiyatlar

metrics_df, ham_fiyatlar = verileri_yukle()

# ==========================================
# YENİ ML MOTORU: RIDGE REGRESYONU
# ==========================================
@st.cache_data
def ml_ile_tahmin_ve_test(fiyat_serisi, hedef_tarih="2026-12-31"):
    df_temiz = fiyat_serisi.dropna().to_frame(name="Fiyat")
    
    # 1. ÖZELLİK MÜHENDİSLİĞİ (Gecikmeler ve Ortalamalar)
    df_temiz['Lag_1'] = df_temiz['Fiyat'].shift(1)
    df_temiz['Lag_3'] = df_temiz['Fiyat'].shift(3)
    df_temiz['Lag_5'] = df_temiz['Fiyat'].shift(5)
    df_temiz['MA_5'] = df_temiz['Fiyat'].rolling(window=5).mean()
    df_temiz['MA_20'] = df_temiz['Fiyat'].rolling(window=20).mean()
    
    df_ml = df_temiz.dropna()
    ozellikler = ['Lag_1', 'Lag_3', 'Lag_5', 'MA_5', 'MA_20']
    
    X = df_ml[ozellikler]
    y = df_ml['Fiyat']
    
    # Verilerin Ölçeklendirilmesi (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. BACKTESTING (Test Seti: Son 6 Ay / 130 İşlem Günü)
    test_gun_sayisi = 130
    split_index = len(df_ml) - test_gun_sayisi
    
    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    # Model Eğitimi
    model_test = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model_test.fit(X_train, y_train)
    test_tahminleri = model_test.predict(X_test)
    
    # Hata Metriklerinin Hesaplanması
    mape = np.mean(np.abs((y_test.values - test_tahminleri) / y_test.values)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, test_tahminleri))
    basari_orani = max(0, 100 - mape)
    
    # 3. GELECEK PROJEKSİYONU (Özyinelemeli Tahmin)
    model_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model_final.fit(X_scaled, y)
    
    son_tarih = df_ml.index[-1]
    hedef = pd.to_datetime(hedef_tarih)
    gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), end=hedef)
    
    tahmin_listesi = []
    aktif_seri = df_ml['Fiyat'].copy()
    
    for tar in gelecek_tarihler:
        lag_1 = aktif_seri.iloc[-1]
        lag_3 = aktif_seri.iloc[-3]
        lag_5 = aktif_seri.iloc[-5]
        ma_5 = aktif_seri.iloc[-5:].mean()
        ma_20 = aktif_seri.iloc[-20:].mean()
        
        X_anlik = pd.DataFrame([[lag_1, lag_3, lag_5, ma_5, ma_20]], columns=ozellikler)
        X_anlik_scaled = scaler.transform(X_anlik)
        
        anlik_pred = model_final.predict(X_anlik_scaled)[0]
        tahmin_listesi.append(anlik_pred)
        aktif_seri[tar] = anlik_pred
        
    tahmin_serisi = pd.Series(tahmin_listesi, index=gelecek_tarihler)
    yil_sonu_fiyat = tahmin_serisi.iloc[-1]
    
    return df_ml['Fiyat'], tahmin_serisi, yil_sonu_fiyat, basari_orani, mape, rmse

# ==========================================
# ŞİRKET İSİMLERİ SÖZLÜĞÜ
# ==========================================
sirket_isimleri = {
    "AKBNK.IS": "Akbank", 
    "ALARK.IS": "Alarko Holding", 
    "ASELS.IS": "Aselsan",
    "ASTOR.IS": "Astor Enerji", 
    "BIMAS.IS": "BİM", 
    "BRSAN.IS": "Borusan Boru",
    "EKGYO.IS": "Emlak Konut", 
    "ENKAI.IS": "Enka İnşaat", 
    "EREGL.IS": "Erdemir",
    "FROTO.IS": "Ford Otosan", 
    "GARAN.IS": "Garanti BBVA", 
    "GUBRF.IS": "Gübre Fabrikaları",
    "HEKTS.IS": "Hektaş", 
    "ISCTR.IS": "İş Bankası", 
    "KCHOL.IS": "Koç Holding",
    "KONTR.IS": "Kontrolmatik", 
    "KRDMD.IS": "Kardemir", 
    "OYAKC.IS": "Oyak Çimento",
    "PETKM.IS": "Petkim", 
    "PGSUS.IS": "Pegasus", 
    "SAHOL.IS": "Sabancı Holding",
    "SASA.IS": "Sasa Polyester", 
    "SISE.IS": "Şişecam", 
    "TCELL.IS": "Turkcell",
    "THYAO.IS": "Türk Hava Yolları", 
    "TOASO.IS": "Tofaş", 
    "TUPRS.IS": "Tüpraş",
    "YKBNK.IS": "Yapı Kredi"
}

# Verinin alfabetik sıralanması ve liste oluşturulması
metrics_df = metrics_df.sort_values(by="Hisse").reset_index(drop=True)
hisse_listesi = metrics_df["Hisse"].tolist()

# Sol Menü (Sidebar) Araçları
st.sidebar.title("Hisse Listesi (A-Z)")
selected_ticker = st.sidebar.radio(
    "Analiz etmek istediğiniz hisseyi seçin:", 
    hisse_listesi,
    format_func=lambda x: f"{x} - {sirket_isimleri.get(x, 'Bilinmiyor')}"
)

hisse_verisi = metrics_df[metrics_df["Hisse"] == selected_ticker].iloc[0]

# ==========================================
# 1. BÖLÜM: FİNANSAL ÖZET
# ==========================================
st.markdown("<hr style='border: 1px solid #444; margin: 15px 0;'>", unsafe_allow_html=True) 
st.subheader(f"🔍 {selected_ticker} ({sirket_isimleri.get(selected_ticker, '')}) Finansal Özet", anchor=False)
col1, col2, col3 = st.columns(3)

# 10 Yıllık Fiyat Verilerinin Hesaplanması
hisse_tum_fiyatlar = ham_fiyatlar[selected_ticker].dropna()
min_fiyat = hisse_tum_fiyatlar.min()
max_fiyat = hisse_tum_fiyatlar.max()

fiyat_aciklamasi = f"Son 10 Yılda Görülen:\n\n⬇️ En Düşük Fiyat: {min_fiyat:.2f} TL\n\n⬆️ En Yüksek Fiyat: {max_fiyat:.2f} TL"
volatilite_aciklamasi = "Volatilite (Oynaklık), hisse fiyatının belirli bir zaman diliminde ortalamadan ne kadar saptığını gösterir."

col1.metric(label="Güncel Fiyat", value=f"{hisse_verisi['Son_Fiyat']:.2f} TL", help=fiyat_aciklamasi)
col2.metric(label="Yıllık Volatilite", value=f"%{hisse_verisi['Volatilite']*100:.2f}", help=volatilite_aciklamasi)
col3.metric(label="Maksimum Düşüş", value=f"%{hisse_verisi['Max_Drawdown']*100:.2f}")

# ==========================================
# 2. BÖLÜM: GÖRSEL RİSK KADRANI VE KISA ÖZET
# ==========================================
st.markdown("<hr style='border: 1px solid #444; margin: 15px 0;'>", unsafe_allow_html=True) 
st.subheader("🤖 Algoritma Geri Bildirimi ve Risk Skalası", anchor=False)

status_class = hisse_verisi["Risk_Sinifi"]
risk_skoru_yuzde = int(hisse_verisi["Risk_Skoru"] * 100)

# Risk Sınıfına Göre Renk ve İkon Belirleme
if risk_skoru_yuzde <= 25:
    bar_color = "#00d4b2" 
    ikon = "🛡️"
elif risk_skoru_yuzde <= 50:
    bar_color = "#29b5e8" 
    ikon = "⚖️"
elif risk_skoru_yuzde <= 75:
    bar_color = "#ffa421" 
    ikon = "⚡"
else:
    bar_color = "#ff4b4b" 
    ikon = "⚠️"

g_col1, g_col2 = st.columns([1, 1])

with g_col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_skoru_yuzde,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "/100", 'font': {'size': 45, 'color': bar_color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "#333",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#00d4b2'}, 
                {'range': [25, 50], 'color': '#29b5e8'}, 
                {'range': [50, 75], 'color': '#ffa421'}, 
                {'range': [75, 100], 'color': '#ff4b4b'} 
            ],
            'threshold': {
                'line': {'color': "white", 'width': 6}, 
                'thickness': 0.8, 
                'value': risk_skoru_yuzde 
            }
        }
    ))
    fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with g_col2:
    st.markdown(f"<h3 style='color:{bar_color};'>{ikon} Sınıf: {status_class}</h3>", unsafe_allow_html=True)
    
    # Dinamik Yatırım Profili Geri Bildirimleri
    if "Çok Güvenilir" in status_class:
        st.markdown(
            "<ul>"
            "<li><b>İstikrar:</b> Endeks içi en yüksek stabilite (Q1).</li>"
            "<li><b>Kriz Direnci:</b> Sert piyasa düşüşlerinde korumacı davranır.</li>"
            "<li><b>Yatırım Profili:</b> Defansif (Düşük Risk) portföyler için idealdir.</li>"
            "</ul>", 
            unsafe_allow_html=True
        )
    elif "Güvenilir" in status_class:
        st.markdown(
            "<ul>"
            "<li><b>İstikrar:</b> Piyasa ile uyumlu, dengeli hareket (Q2).</li>"
            "<li><b>Kriz Direnci:</b> Ortalama düzeyde tepki verir.</li>"
            "<li><b>Yatırım Profili:</b> Uzun vadeli ana yatırımlar için makuldür.</li>"
            "</ul>", 
            unsafe_allow_html=True
        )
    elif "Az Güvenilir" in status_class:
        st.markdown(
            "<ul>"
            "<li><b>İstikrar:</b> Ortalamanın üzerinde oynaklık (Q3).</li>"
            "<li><b>Kriz Direnci:</b> Ani ve sert fiyat düzeltmelerine açıktır.</li>"
            "<li><b>Yatırım Profili:</b> Getiri potansiyeli yüksek, stres toleransı gerektirir.</li>"
            "</ul>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<ul>"
            "<li><b>İstikrar:</b> Günlük dalgalanmaları çok yüksektir (Q4).</li>"
            "<li><b>Kriz Direnci:</b> Tarihsel olarak çok derin değer kayıpları yaşanmıştır.</li>"
            "<li><b>Yatırım Profili:</b> Sadece agresif ve yüksek risk seven profiller içindir.</li>"
            "</ul>", 
            unsafe_allow_html=True
        )

# ==========================================
# 3. BÖLÜM: DİNAMİK BİLGİ KARTLARI
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Güncel Volatilite Sınırları", anchor=False)

summary_df = metrics_df.groupby("Risk_Sinifi").agg(
    Min_Volatilite=('Volatilite', lambda x: f"%{x.min()*100:.1f}"),
    Max_Volatilite=('Volatilite', lambda x: f"%{x.max()*100:.1f}")
).reset_index()

c1, c2, c3, c4 = st.columns(4)
sutunlar = [c1, c2, c3, c4]

sinif_ayarlari = {
    "Çok Güvenilir (Defansif)": {"renk": "#00d4b2", "ikon": "🛡️", "isim": "Çok Güvenilir"},
    "Güvenilir (Dengeli)": {"renk": "#29b5e8", "ikon": "⚖️", "isim": "Güvenilir"},
    "Az Güvenilir (Dinamik)": {"renk": "#ffa421", "ikon": "⚡", "isim": "Az Güvenilir"},
    "Güvenilmez (Agresif)": {"renk": "#ff4b4b", "ikon": "⚠️", "isim": "Güvenilmez"}
}

sira = [
    "Çok Güvenilir (Defansif)", 
    "Güvenilir (Dengeli)", 
    "Az Güvenilir (Dinamik)", 
    "Güvenilmez (Agresif)"
]

for i, sinif in enumerate(sira):
    if sinif in summary_df['Risk_Sinifi'].values:
        row = summary_df[summary_df['Risk_Sinifi'] == sinif].iloc[0]
        alt_sinir = row['Min_Volatilite']
        ust_sinir = row['Max_Volatilite']
    else:
        alt_sinir = "-"
        ust_sinir = "-"
    
    renk = sinif_ayarlari[sinif]["renk"]
    ikon = sinif_ayarlari[sinif]["ikon"]
    isim = sinif_ayarlari[sinif]["isim"]
    
    html_card = f"""
    <div style="background-color: #2b2b2b; padding: 25px 15px; border-radius: 12px; border-top: 6px solid {renk}; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <h4 style="color: {renk}; margin-top: 0; font-size: 32px;">{ikon}</h4>
        <h5 style="color: white; margin: 15px 0; font-size: 24px;">{isim}</h5>
        <p style="color: #ccc; font-size: 24px; font-weight: bold; margin-bottom: 0;">{alt_sinir} - {ust_sinir}</p>
    </div>
    """
    with sutunlar[i]:
        st.markdown(html_card, unsafe_allow_html=True)

# ==========================================
# 4. BÖLÜM: ML TAHMİN PROJEKSİYONU
# ==========================================
st.markdown("<hr style='border: 1px solid #444; margin: 15px 0;'>", unsafe_allow_html=True) 
st.subheader(f"🔮 {selected_ticker} Gelişmiş Ridge Regresyon Projeksiyonu", anchor=False)

gecmis_veri, gelecek_tahmin, yil_sonu_hedef, basari_orani, mape_degeri, rmse_degeri = ml_ile_tahmin_ve_test(ham_fiyatlar[selected_ticker])

if yil_sonu_hedef is not None:
    guncel_fiyat = hisse_verisi['Son_Fiyat']
    beklenen_getiri_orani = ((yil_sonu_hedef - guncel_fiyat) / guncel_fiyat) * 100
    
    # 3'lü Metrik Tasarımı (Doğruluk Oranı ayrı bir kart, açıklaması pop-up içinde)
    t_col1, t_col2, t_col3 = st.columns(3)
    
    t_col1.metric(
        label="Mevcut Fiyat (Haziran 2026)", 
        value=f"{guncel_fiyat:.2f} TL"
    )
    
    t_col2.metric(
        label="Modelin 2026 Yıl Sonu Öngörüsü", 
        value=f"{yil_sonu_hedef:.2f} TL", 
        delta=f"%{beklenen_getiri_orani:.2f} Trend Değişimi"
    )
    
    basari_aciklamasi = "Bu makine öğrenmesi algoritması, 2026 yıl sonu hedefini hesaplamadan önce son 6 aylık piyasa verileri üzerinde kendini test etmiş ve bu doğruluk oranına ulaşmıştır."
    
    t_col3.metric(
        label="🎯 Model Doğruluk Oranı", 
        value=f"%{basari_orani:.1f}", 
        help=basari_aciklamasi
    )
    
    # Grafik Çizimi
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=gecmis_veri.index, 
        y=gecmis_veri.values, 
        mode='lines', 
        name='Tarihsel Fiyat', 
        line=dict(color=bar_color, width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=gelecek_tahmin.index, 
        y=gelecek_tahmin.values, 
        mode='lines', 
        name='Ridge Yıl Sonu Trendi', 
        line=dict(color='#9b59b6', width=5, dash='dash')
    ))
    
    fig.update_layout(
        font=dict(size=28), 
        xaxis_title="<b>Tarih</b>", 
        yaxis_title="<b>Fiyat (TL)</b>",
        margin=dict(l=50, r=50, t=100, b=50), 
        height=800,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0, 
            font=dict(size=32)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Yetersiz zaman serisi verisi nedeniyle tahmin modeli oluşturulamadı.")

# ==========================================
# 5. BÖLÜM: YASAL UYARI VE BİLGİLENDİRME
# ==========================================
st.sidebar.markdown("<hr style='border: 1px solid #444; margin: 20px 0;'>", unsafe_allow_html=True)
st.sidebar.warning(
    "⚠️ **Sorumluluk Reddi Beyanı:**\n\n"
    "Bu portalda sunulan analizler, risk skorları ve makine öğrenmesi tahmin modelleri "
    "tamamen tarihsel verilere dayalı **akademik** amaçlı çıktılardır. "
    "Yatırımcılara tarafsız istatistiksel bilgi sağlamak amacıyla geliştirilmiş olup, "
    "kesinlikle **Yatırım Tavsiyesi Değildir (YTD)**. "
    "Karar mekanizması tamamen kullanıcının hür iradesine ait olup, sistem hiçbir manipülatif yönlendirme içermez."
)
import pandas as pd
import numpy as np

def risk_metriklerini_hesapla(excel_dosyasi="bist30_10yillik_fiyatlar.xlsx"):
    # 1. Ham veriyi Excel'den oku (İnternet bağlantısına gerek yok!)
    df = pd.read_excel(excel_dosyasi, index_col=0) # Tarih sütununu indeks yap
    
    metrics_list = []

    # 2. Her bir hisse için döngü başlat
    for ticker in df.columns:
        # Hisselerin boş (NaN) günlerini atla
        fiyat_serisi = df[ticker].dropna() 
        if len(fiyat_serisi) == 0:
            continue

        # MATEMATİKSEL HESAPLAMALAR
        # a. Logaritmik Getiri
        log_getiri = np.log(fiyat_serisi / fiyat_serisi.shift(1))

        # b. Yıllıklandırılmış Tarihsel Volatilite (252 işlem günü)
        volatilite = log_getiri.std() * np.sqrt(252)

        # c. Maksimum Düşüş (Max Drawdown) - En yüksek zirveden dibe kayıp
        rolling_max = fiyat_serisi.cummax()
        drawdown = (fiyat_serisi - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Son güncel fiyat
        son_fiyat = fiyat_serisi.iloc[-1]

        # Sonuçları listeye ekle
        metrics_list.append({
            "Hisse": ticker,
            "Son_Fiyat": son_fiyat,
            "Volatilite": volatilite,
            "Max_Drawdown": max_dd
        })

    # Listeyi yapısal bir DataFrame'e dönüştür
    metrics_df = pd.DataFrame(metrics_list)

    # 3. RİSK SKORU VE NORMALİZASYON (%60 Volatilite, %40 Max Drawdown)
    # Metrikleri 0 ile 1 arasına sıkıştırıyoruz (Min-Max Scaling)
    metrics_df["Norm_Vol"] = (metrics_df["Volatilite"] - metrics_df["Volatilite"].min()) / (metrics_df["Volatilite"].max() - metrics_df["Volatilite"].min())
    
    # Max Drawdown negatif bir değerdir, risk hesaplarken mutlak değerini (abs) alıyoruz
    abs_dd = metrics_df["Max_Drawdown"].abs()
    metrics_df["Norm_DD"] = (abs_dd - abs_dd.min()) / (abs_dd.max() - abs_dd.min())

    metrics_df["Risk_Skoru"] = (metrics_df["Norm_Vol"] * 0.6) + (metrics_df["Norm_DD"] * 0.4)

    # 4. DİNAMİK EŞİKLER VE SINIFLANDIRMA (Quartile / Çeyreklik)
    q1 = metrics_df["Risk_Skoru"].quantile(0.25)
    q2 = metrics_df["Risk_Skoru"].quantile(0.50)
    q3 = metrics_df["Risk_Skoru"].quantile(0.75)

    def siniflandir(skor):
        if skor <= q1:
            return "Çok Güvenilir (Defansif)"
        elif skor <= q2:
            return "Güvenilir (Dengeli)"
        elif skor <= q3:
            return "Az Güvenilir (Dinamik)"
        else:
            return "Güvenilmez (Agresif)"

    metrics_df["Risk_Sinifi"] = metrics_df["Risk_Skoru"].apply(siniflandir)
    
    # Okunabilirlik için skora göre sırala (En düşük riskten en yükseğe)
    metrics_df = metrics_df.sort_values(by="Risk_Skoru").reset_index(drop=True)

    return metrics_df, df

# BU DOSYA DOĞRUDAN ÇALIŞTIRILIRSA AŞAĞIDAKİ TEST KODU ÇALIŞIR
if __name__ == "__main__":
    print("Veriler Excel'den okunuyor ve hesaplamalar yapılıyor...\n")
    
    sonuclar_tablosu, ham_fiyatlar = risk_metriklerini_hesapla()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("✅ HESAPLAMA TAMAMLANDI!\n")
    print("--- EN GÜVENİLİR (DEFANSİF) 5 HİSSE ---")
    print(sonuclar_tablosu[['Hisse', 'Volatilite', 'Max_Drawdown', 'Risk_Skoru', 'Risk_Sinifi']].head())
    
    print("\n--- EN RİSKLİ (AGRESİF) 5 HİSSE ---")
    print(sonuclar_tablosu[['Hisse', 'Volatilite', 'Max_Drawdown', 'Risk_Skoru', 'Risk_Sinifi']].tail())
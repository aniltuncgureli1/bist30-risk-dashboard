import yfinance as yf
import pandas as pd

bist30_hisseler = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", 
    "BRSAN.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", 
    "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", 
    "KONTR.IS", "KRDMD.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", 
    "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", 
    "TOASO.IS", "TUPRS.IS", "YKBNK.IS"
]

print("Yahoo Finance sunucularına bağlanılıyor...")
print("BIST 30 için son 10 yıllık veriler çekiliyor, lütfen bekleyin...\n")

veri = yf.download(bist30_hisseler, period="10y", progress=False)

# Sütun adı kontrolü yapan akıllı filtre
if 'Adj Close' in veri.columns:
    fiyatlar = veri['Adj Close']
else:
    fiyatlar = veri['Close'] # yfinance auto-adjust yaptıysa burası çalışır

print("✅ Veriler başarıyla çekildi!")
print(f"Toplam {fiyatlar.shape[0]} işlem günü ve {fiyatlar.shape[1]} hisse senedi verisi alındı.\n")

print(fiyatlar.head())

# BÜTÜN VERİYİ EXCEL OLARAK KAYDETME ADIMI
fiyatlar.to_excel("bist30_10yillik_fiyatlar.xlsx")
print("\n📊 Excel dosyası başarıyla oluşturuldu: 'bist30_10yillik_fiyatlar.xlsx'")
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import uvicorn

app = FastAPI(title="Gerçek ML Destekli SAP Fiyat Anomali Servisi")

X_train = np.array([
    [100], [120], [105], [110], [115], [130], [95], [102],
    [450], [480], [500], [460], [490], [510], [475]
])

# Anomali Tespiti için Machine Learning Algoritması: Isolation Forest
ml_model = IsolationForest(contamination=0.1, random_state=42)
ml_model.fit(X_train)  

# 2. SAP'DEN GELEN İSTEKLERİ KARŞILAMA
class MaterialRequest(BaseModel):
    matnr: str
    price: float
    currency: str

@app.get("/")
def root():
    return {
        "service": "SAP MM AI Price Analyzer",
        "status": "ok",
        "endpoint": "/analyze-price"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/analyze-price")
def analyze_price(data: MaterialRequest):
    
    # SAP'den gelen fiyatı Yapay Zeka Modele atıyoruz:
    yeni_fiyat = np.array([[data.price]])
    
    # MODEL TAHMİNİ (PREDICTION): 
    # Yapay zeka geçmiş verilerine bakarak karar veriyor (1: Normal, -1: Anomali)
    tahmin = ml_model.predict(yeni_fiyat)
    skor = ml_model.score_samples(yeni_fiyat) # Anormallik derecesi
    
    if tahmin[0] == -1:
        is_anomaly = True
        message = f"DİKKAT: Yapay Zeka bu fiyatı ({data.price} {data.currency}) geçmiş desenlere aykırı buldu!"
    else:
        is_anomaly = False
        message = "Fiyat Yapay Zeka modelinin güvenli alanında."

    return {
        "material_id": data.matnr,
        "price": data.price,
        "is_anomaly": is_anomaly,
        "ai_analysis_message": message
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

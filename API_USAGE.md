# API Usage - House Price Linear Regression

## Prasyarat
- Project path mengandung spasi (`ML Project`), jadi gunakan path yang diapit tanda kutip.
- Disarankan jalankan semua command dari root project: `House_Price_Regression`.

## 1) Install dependency (venv)
```bash
"/Users/mac/Documents/coding-mengcoding/Python/ML Project/House_Price_Regression/.venv/bin/python" -m pip install -r requirements.txt kagglehub
```

## 2) Generate artifact model
API membutuhkan 2 file artifact:
- `house_price_linear_regression.pkl`
- `scaler_linear_regression.pkl`

Generate dengan script training:
```bash
"/Users/mac/Documents/coding-mengcoding/Python/ML Project/House_Price_Regression/.venv/bin/python" train_model.py
```

Verifikasi file:
```bash
ls -la
```

## 3) Jalankan API
```bash
"/Users/mac/Documents/coding-mengcoding/Python/ML Project/House_Price_Regression/.venv/bin/python" -m uvicorn app:app --reload
```

Default server:
- Base URL: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## 4) Endpoint
### GET `/`
Cek API aktif.

### GET `/health`
Cek status readiness model/scaler.

Contoh:
```bash
curl http://127.0.0.1:8000/health
```

Response siap pakai:
```json
{
  "status": "ready",
  "model_loaded": true,
  "artifact_files": {
    "house_price_linear_regression.pkl": true,
    "scaler_linear_regression.pkl": true
  }
}
```

### POST `/predict`
Prediksi harga rumah dari 7 fitur.

Body JSON:
```json
{
  "Square_Footage": 2200,
  "Num_Bedrooms": 4,
  "Num_Bathrooms": 3,
  "Year_Built": 2015,
  "Lot_Size": 5500,
  "Garage_Size": 2,
  "Neighborhood_Quality": 8
}
```

Contoh request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Square_Footage": 2200,
    "Num_Bedrooms": 4,
    "Num_Bathrooms": 3,
    "Year_Built": 2015,
    "Lot_Size": 5500,
    "Garage_Size": 2,
    "Neighborhood_Quality": 8
  }'
```

Contoh response:
```json
{
  "predicted_house_price": 345678.9,
  "currency": "USD",
  "model": "LinearRegression"
}
```

## 5) Troubleshooting
### `503 Service Unavailable` pada `/predict`
Penyebab: artifact model/scaler belum ada atau belum terbaca saat startup API.

Solusi:
1. Jalankan `train_model.py`.
2. Pastikan 2 file `.pkl` ada di root project.
3. Restart uvicorn.
4. Cek `/health` sampai status `ready`.

### Prediksi terlihat tidak masuk akal
Input bisa berada di luar distribusi data training (out-of-distribution), misalnya `Square_Footage` atau `Lot_Size` terlalu ekstrem.
Gunakan rentang input yang realistis sesuai data training.

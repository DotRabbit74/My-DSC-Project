# My-DSC-Project

Deep Scene Curve (DSC) 水下影像增強模型復刻與改良專案。

## 專案結構

- `app.py`: Streamlit 應用程式主程式。
- `model_dsc.py`: 模型定義檔 (包含 Network 類別)。
- `loss.py`: 損失函數定義。
- `requirements.txt`: 專案依賴套件清單。
- `weights/`: 存放預訓練權重。
  - `original.pth`: 原始 Sigmoid 版本權重。
  - `softsign.pth`: 改良 Softsign 版本權重。

## 使用方法

1. 安裝依賴套件:
   ```bash
   pip install -r requirements.txt
   ```

2. 執行 Streamlit 應用程式:
   ```bash
   streamlit run app.py
   ```

## 功能

- 比較原始 Sigmoid 方法與 Softsign 方法的效能與畫質。
- 支援上傳圖片進行測試。
- 顯示推論時間與加速幅度。

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import time
import os
import sys

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="Deep Scene Curve Demo",
    page_icon="🌊",
    layout="wide"
)

# --- 2. 檢查並匯入模型 ---
# 為了防止部署時路徑問題，將當前目錄加入 path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # 假設您的模型檔名為 model_dsc.py
    from model_dsc import Network
except ImportError:
    st.error("❌ 找不到 `model_dsc.py`。請確保此檔案已上傳至 GitHub 儲存庫的根目錄。")
    st.stop()

# --- 3. 設定執行裝置 (Streamlit Cloud 通常是 CPU) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(weights_path, mode):
    """
    載入模型權重。
    使用 @st.cache_resource 避免每次網頁刷新都重新讀取模型。
    """
    # 1. 檢查檔案是否存在
    if not os.path.exists(weights_path):
        return None

    # 2. 初始化模型
    try:
        model = Network(mode=mode)
    except TypeError:
        st.error(f"❌ 模型初始化失敗：Network 類別似乎不支援 mode='{mode}' 參數。")
        return None

    # 3. 載入權重 (強制 map_location 到正確裝置，避免 GPU/CPU 衝突)
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"⚠️ 權重檔損毀或不相容 ({weights_path}): {e}")
        return None

    model.to(device)
    model.eval()
    return model

def process_image(model, image):
    """
    影像推論與計時 (支援高解析度)
    """
    # 取得原始尺寸
    w, h = image.size
    
    # --- 安全機制：限制最大邊長 ---
    # Streamlit Cloud 免費版記憶體有限，若圖片過大(如 4K)可能會 OOM (Out of Memory)
    # 我們設定一個上限 (例如 1024 或 1280)，超過就等比例縮小
    max_size = 1024
    scale_factor = 1.0
    
    if max(w, h) > max_size:
        scale_factor = max_size / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        # 使用高品質縮放
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 轉為 Tensor (不強制 Resize 到 256x256 了)
    transform = T.Compose([
        T.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 確保維度是 16 的倍數 (某些卷積網路對尺寸有要求，避免 padding 錯誤)
    # 雖然 DSC 架構通常不強制，但這是一個保險做法
    # 這裡我們先直接丟進去，若報錯再調整
    
    start_time = time.time()
    try:
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
    except RuntimeError as e:
        # 捕捉記憶體不足錯誤
        if "out of memory" in str(e):
            return image, 0.0  # 回傳原圖並報錯 (可以在外層處理)
        raise e
            
    end_time = time.time()
    
    # 轉回 PIL 圖片
    output = torch.clamp(output, 0, 1).squeeze(0).cpu()
    output_img = T.ToPILImage()(output)
    
    return output_img, end_time - start_time

# --- 4. 側邊欄與模型載入 ---
st.sidebar.title("🌊 設定面板")
st.sidebar.caption(f"目前運行裝置: `{device}`")
st.sidebar.info("說明：此應用程式比較原始 Sigmoid 方法與嘗試版 Softsign 方法在水下影像增強的表現。")

# 定義權重路徑 (相對路徑，適配 GitHub 結構)
PATH_ORIGINAL = "weights/original.pth"
PATH_SOFTSIGN = "weights/softsign.pth"

# 載入模型
model_orig = load_model(PATH_ORIGINAL, mode='original')
model_soft = load_model(PATH_SOFTSIGN, mode='softsign')

# --- 5. 主畫面邏輯 ---
st.title("🌊 Deep Scene Curve (DSC) - Model Comparison")
st.markdown("""
本專案復刻並改良了 **Deep Scene Curve** 水下影像增強模型。
我們提出了基於 **Softsign** 的快速曲線估計方法，看是否能提升推論速度並改善梯度傳遞。
""")

uploaded_file = st.file_uploader("📂 請上傳水下圖片 (jpg, png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    # 選項分頁
    tab1, tab2 = st.tabs(["🔍 單一模型分析", "⚡ A/B 效能對決"])

    with tab1:
        st.subheader("單一模型詳細測試")
        option = st.radio("選擇模型版本", ["Original (Sigmoid)", "Modified (Softsign)"], horizontal=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原始輸入", use_container_width=True)
            
        with col2:
            target_model = model_orig if option == "Original (Sigmoid)" else model_soft
            
            if target_model:
                res, t = process_image(target_model, image)
                st.image(res, caption=f"增強結果 ({option})", use_container_width=True)
                st.success(f"⏱️ 推論時間: {t*1000:.2f} ms")
                
                # 顯示數學公式
                if option == "Original (Sigmoid)":
                    st.latex(r"\mathcal{F}(x) = \frac{1}{1+e^{-(\alpha x + \beta)}}")
                    st.caption("原始論文使用 Standard Sigmoid，包含指數運算。")
                else:
                    st.latex(r"\mathcal{F}(x) = 0.5 \times \left( \frac{\alpha x + \beta}{1 + |\alpha x + \beta|} + 1 \right)")
                    st.caption("嘗試版使用 Rescaled Softsign，僅需代數運算，速度更快。")
            else:
                st.warning(f"⚠️ 找不到權重檔，請確認 `{PATH_ORIGINAL}` 或 `{PATH_SOFTSIGN}` 是否存在於 GitHub。")

    with tab2:
        st.subheader("⚡ 效能與畫質並列比較")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original Input**")
            st.image(image, use_container_width=True)
            
        t_o, t_s = 0, 0
        
        with c2:
            st.markdown("**Original (Sigmoid)**")
            if model_orig:
                res_o, t_o = process_image(model_orig, image)
                st.image(res_o, use_container_width=True)
                st.info(f"{t_o*1000:.2f} ms")
            else:
                st.error("Missing Weights")
                
        with c3:
            st.markdown("**Modified (Softsign)**")
            if model_soft:
                res_s, t_s = process_image(model_soft, image)
                st.image(res_s, use_container_width=True)
                st.info(f"{t_s*1000:.2f} ms")
            else:
                st.error("Missing Weights")
        
        # 結論
        if t_o > 0 and t_s > 0:
            st.markdown("---")
            speedup = (t_o - t_s) / t_o * 100
            if speedup > 0:
                st.metric(label="Softsign 加速幅度", value=f"{speedup:.2f}%", delta="Faster")
            else:
                st.metric(label="速度差異", value=f"{abs(speedup):.2f}%")

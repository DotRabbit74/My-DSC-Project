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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from model_dsc import Network
except ImportError:
    st.error("❌ 找不到 `model_dsc.py`。請確保此檔案已上傳至 GitHub 儲存庫的根目錄。")
    st.stop()

# --- 3. 設定執行裝置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(weights_path, mode):
    """ 載入模型權重 """
    if not os.path.exists(weights_path):
        return None

    try:
        model = Network(mode=mode)
    except TypeError:
        st.error(f"❌ 模型初始化失敗：Network 類別似乎不支援 mode='{mode}' 參數。")
        return None

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
    影像推論與計時 (支援高解析度 + 記憶體保護) 
    """
    w, h = image.size
    
    # --- 安全機制：限制最大邊長 ---
    # 防止 4K 圖在 Streamlit Cloud 免費版 OOM (Out Of Memory)
    max_size = 1280
    if max(w, h) > max_size:
        scale_factor = max_size / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 轉為 Tensor (不強制 Resize 到 256x256，保持畫質)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    start_time = time.time()
    try:
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
    except RuntimeError as e:
        if "out of memory" in str(e):
            return image, 0.0 # 記憶體不足時回傳原圖
        raise e
            
    end_time = time.time()
    
    # 轉回 PIL
    output = torch.clamp(output, 0, 1).squeeze(0).cpu()
    output_img = T.ToPILImage()(output)
    
    return output_img, end_time - start_time

# --- 4. 側邊欄設定 ---
st.sidebar.title("🌊 設定面板")
st.sidebar.caption(f"Device: `{device}`")
st.sidebar.info("說明：此應用程式比較原始 Sigmoid 方法與改良版 Softsign 方法在水下影像增強的表現。")

# 定義權重路徑
PATH_ORIGINAL = "weights/original.pth"
PATH_SOFTSIGN = "weights/softsign.pth"

# 載入模型
model_orig = load_model(PATH_ORIGINAL, mode='original')
model_soft = load_model(PATH_SOFTSIGN, mode='softsign')

# --- 5. 主畫面邏輯 ---
st.title("🌊 Deep Scene Curve (DSC) - Model Comparison")
st.markdown("""
本專案復刻並改良了 **Deep Scene Curve** 水下影像增強模型。
使用 **Softsign** 曲線估計方法，以提升推論速度並改善梯度傳遞。
""")

# --- [新增功能] 圖片來源選擇邏輯 ---
image = None
uploaded_file = st.file_uploader("📂 上傳圖片 (或使用下方範例)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 優先使用上傳的圖片
    image = Image.open(uploaded_file).convert('RGB')
else:
    # 若無上傳，檢查 samples 資料夾
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        # 取得資料夾內所有圖片
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_files:
            # 顯示下拉選單
            selected_sample = st.selectbox(
                "🖼️ 沒有圖片嗎？選擇一張範例圖片來測試：",
                sample_files,
                index=0
            )
            # 載入選擇的範例圖
            image_path = os.path.join(sample_dir, selected_sample)
            image = Image.open(image_path).convert('RGB')
    
# --- 6. 展示與推論 ---
if image:
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
                with st.spinner("正在增強中..."):
                    res, t = process_image(target_model, image)
                
                st.image(res, caption=f"增強結果 ({option})", use_container_width=True)
                st.success(f"⏱️ 推論時間: {t*1000:.2f} ms")
                
                if option == "Original (Sigmoid)":
                    st.latex(r"\mathcal{F}(x) = \frac{1}{1+e^{-(\alpha x + \beta)}}")
                else:
                    st.latex(r"\mathcal{F}(x) = 0.5 \times \left( \frac{\alpha x + \beta}{1 + |\alpha x + \beta|} + 1 \right)")
            else:
                st.warning(f"⚠️ 找不到權重檔，請確認 GitHub 上是否有 `{PATH_ORIGINAL}` 或 `{PATH_SOFTSIGN}`。")

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
        
        if t_o > 0 and t_s > 0:
            st.markdown("---")
            speedup = (t_o - t_s) / t_o * 100
            if speedup > 0:
                st.metric(label="Softsign 加速幅度", value=f"{speedup:.2f}%", delta="Faster")
            else:
                st.metric(label="速度差異", value=f"{abs(speedup):.2f}%")

else:
    # 若沒有上傳也沒有範例圖
    st.info("👋 請上傳圖片以開始測試！")

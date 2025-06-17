import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
import plotly.express as px

# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="YOLOv8 Feature Visualizer", layout="centered")

# 🔍 Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt").model

model = load_model()

# 🚀 App title and description
st.title("🎯 YOLOv8 Feature Map Visualizer – WhiteBox CNN")
st.markdown("Upload an image and explore internal CNN layers using **fancy interactive plots**.")

# 📤 Upload image
uploaded_file = st.file_uploader(
    "Upload an image (JPG, PNG, BMP, TIFF, WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"]
)

if uploaded_file:
    # Load and show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # 🔍 Select layer group (Backbone / Neck / Head)
    layer_dict = {
        "Backbone (Conv/C2f/SPPF)": list(range(0, 6)),
        "Neck (PANet)": list(range(6, 9)),
        "Head (Detection)": [9]
    }
    layer_type = st.selectbox("🧱 Select Layer Group", list(layer_dict.keys()))
    selected_layers = layer_dict[layer_type]

    # 🧪 Preprocess image
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    # 🎯 Hook to capture feature maps
    feature_maps = {}
    def get_hook(name):
        def hook_fn(module, input, output):
            feature_maps[name] = output
        return hook_fn

    # 🔗 Register hooks
    hooks = []
    for i in selected_layers:
        hook = model.model[i].register_forward_hook(get_hook(f"layer_{i}"))
        hooks.append(hook)

    # 🔄 Run forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # 🧹 Remove hooks
    for h in hooks:
        h.remove()

    # 📊 Visualize feature maps
    for layer_name, fmap in feature_maps.items():
        fmap = fmap.squeeze(0)  # Remove batch dimension
        st.markdown(f"### 🧬 {layer_name} – Shape: `{tuple(fmap.shape)}`")

        num_channels = st.slider(
            f"📊 Channels to visualize in {layer_name}",
            min_value=1,
            max_value=min(8, fmap.shape[0]),
            value=4
        )

        for i in range(num_channels):
            fmap_np = fmap[i].detach().cpu().numpy()
            fig = px.imshow(
                fmap_np,
                color_continuous_scale='viridis',
                title=f"Channel {i} – {layer_name}",
                labels=dict(color="Activation")
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬆️ Upload an image to explore YOLOv8's internal feature maps.")

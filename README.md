# CNN-Yolov8-WhiteBox

# 🧠 YOLOv8 Feature Map Visualizer – WhiteBox CNN

**Explore the inner workings of YOLOv8!**  
This interactive Streamlit app lets you upload an image and visualize feature maps from YOLOv8’s backbone, neck, and head — turning it from a black-box into a white-box CNN you can understand.

![demo](https://github.com/your-username/your-repo-name/blob/main/assets/demo.gif)

---

## 🔍 Features

- 📤 Upload any image (JPG, PNG, BMP, TIFF, WEBP)
- 🧱 Select YOLOv8 layers: Backbone, Neck, or Head
- 🧬 Visualize intermediate **feature maps** using interactive Plotly graphs
- 🎛 Customize how many channels to visualize
- 💡 Built using `Ultralytics`, `PyTorch`, and `Streamlit`

---

## 🛠 Tech Stack

| Tool         | Purpose                        |
|--------------|--------------------------------|
| `YOLOv8`     | Pretrained detection model     |
| `Torch`      | Tensor operations & hooks      |
| `Streamlit`  | Interactive web UI             |
| `Plotly`     | Fancy, zoomable visualizations |
| `Pillow`     | Image processing               |

---

## 🚀 Try It Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/yolov8-visualizer.git
   cd yolov8-visualizer

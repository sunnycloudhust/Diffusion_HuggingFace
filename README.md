# Huấn luyện Mô hình Diffusion (DDPM) với Hugging Face

Dự án này chứa mã nguồn để xây dựng và huấn luyện một mô hình **Denoising Diffusion Probabilistic Models (DDPM)** từ con số 0. Quá trình huấn luyện sử dụng thư viện `diffusers` của Hugging Face và được thực hiện trên bộ dữ liệu ảnh màu **CIFAR-10** (độ phân giải 32x32).

## 📂 Cấu trúc Dự án

Dự án được chia nhỏ thành các module để dễ quản lý:

* **`data.py`**: Chịu trách nhiệm tải bộ dữ liệu CIFAR-10 từ Hugging Face Datasets, áp dụng các kỹ thuật tiền xử lý (như Resize, RandomHorizontalFlip, Normalize) và đóng gói vào PyTorch `DataLoader`.
* **`model.py`**: Chứa phần lõi của thuật toán bao gồm:
  * Khởi tạo kiến trúc mạng Nơ-ron `UNet2DModel`.
  * Thiết lập bộ lập lịch thêm/giải nhiễu `DDPMScheduler` (sử dụng lịch trình `squaredcos_cap_v2`).
  * Vòng lặp huấn luyện chính (Training Loop) tính toán Loss và cập nhật trọng số.
* **`.gitignore`**: Cấu hình bỏ qua các file tạm của Python, môi trường ảo và các file trọng số mô hình lớn khi đẩy code lên GitHub.

## 🛠️ Yêu cầu Hệ thống (Requirements)

Để chạy dự án này, bạn cần cài đặt Python 3.8+ và các thư viện sau:

```bash
pip install torch torchvision
pip install diffusers datasets

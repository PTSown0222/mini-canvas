FROM python:3.12-slim

# Ngăn Python tạo bytecode .pyc và bật unbuffered log
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    U2NET_HOME=/app/models

# Cài đặt thư viện C/C++ runtime cho OpenCV và công cụ tải uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv package manager trực tiếp từ binary chính thức
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy các file cấu hình dependency để tận dụng Docker layer cache
COPY pyproject.toml uv.lock ./

# Đồng bộ cài đặt dependencies hệ thống bằng uv (không tạo .venv bên trong image)
RUN uv sync --frozen --no-dev --no-install-project

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Khai báo port mặc định của Streamlit
EXPOSE 8501

# Lệnh khởi chạy ứng dụng
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
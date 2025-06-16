# Use a imagem base fornecida
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie SEU script para a imagem (não copie o modelo!)
COPY starter.py .

# Instale dependências se necessário (crie requirements.txt primeiro)
# RUN pip install -r requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Comando para executar seu script
CMD ["python", "starter.py", "--year", "2023", "--month", "5"]
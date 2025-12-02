# Real-ESRGAN API Server

REST API сервер для обработки изображений с помощью Real-ESRGAN.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
python setup.py develop
```

2. Убедитесь, что у вас установлен PyTorch с поддержкой CUDA (если используете GPU):
```bash
# Для CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Запуск сервера

### Вариант 1: Прямой запуск Python скрипта
```bash
python api_server.py
```

### Вариант 2: Использование uvicorn напрямую
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Вариант 3: Использование скрипта запуска
```bash
chmod +x start_api.sh
./start_api.sh 8000 0.0.0.0
```

Сервер будет доступен по адресу: `http://localhost:8000`

## API Документация

После запуска сервера доступна интерактивная документация:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### GET `/`
Информация о API и доступных endpoints.

### GET `/health`
Проверка состояния сервера. Возвращает статус и информацию о GPU.

### GET `/models`
Список доступных моделей.

### POST `/upscale`
Обработка изображения.

**Параметры (multipart/form-data):**
- `file` (обязательный): Файл изображения
- `model_name` (опционально, по умолчанию: `RealESRGAN_x4plus`): Название модели
  - `RealESRGAN_x4plus` - общая модель x4
  - `RealESRNet_x4plus` - общая модель x4 (без GAN)
  - `RealESRGAN_x4plus_anime_6B` - модель для аниме x4
  - `RealESRGAN_x2plus` - общая модель x2
  - `realesr-animevideov3` - модель для аниме видео
  - `realesr-general-x4v3` - компактная общая модель x4
- `scale` (опционально, по умолчанию: 4.0): Финальный масштаб увеличения
- `face_enhance` (опционально, по умолчанию: false): Улучшение лиц с помощью GFPGAN
- `tile` (опционально, по умолчанию: 0): Размер тайла для больших изображений
  - `0` = автоматическое определение (включается для изображений > 500px)
  - `512` = автоматически для GPU с 12GB+ памяти (RTX 3080 Ti, 3090, etc.)
  - `400` = автоматически для GPU с 8GB+ памяти
  - `200` = автоматически для GPU с 4GB памяти
  - Явное указание значения отключает автоопределение
- `denoise_strength` (опционально, по умолчанию: 0.5): Сила шумоподавления для `realesr-general-x4v3` (0-1)
- `fp32` (опционально, по умолчанию: false): Использовать полную точность (медленнее, но точнее)
- `gpu_id` (опционально): ID GPU устройства (None = автоматический выбор)

**Ответ:** Обработанное изображение в формате файла.

## Примеры использования

### cURL
```bash
curl -X POST "http://localhost:8000/upscale" \
  -F "file=@input.jpg" \
  -F "model_name=RealESRGAN_x4plus" \
  -F "scale=4.0" \
  -F "face_enhance=false" \
  -o output.jpg
```

### Python
```python
import requests

url = "http://localhost:8000/upscale"
files = {"file": open("input.jpg", "rb")}
data = {
    "model_name": "RealESRGAN_x4plus",
    "scale": 4.0,
    "face_enhance": False
}

response = requests.post(url, files=files, data=data)
with open("output.jpg", "wb") as f:
    f.write(response.content)
```

### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_name', 'RealESRGAN_x4plus');
formData.append('scale', '4.0');
formData.append('face_enhance', 'false');

fetch('http://localhost:8000/upscale', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'upscaled.jpg';
    a.click();
});
```

## Производительность

- Модели кэшируются в памяти после первой загрузки
- **Автоматический тайлинг**: Для изображений > 500px по любой стороне тайлинг включается автоматически:
  - GPU с 12GB+ памяти (RTX 3080 Ti, 3090): tile=512
  - GPU с 8GB+ памяти: tile=400
  - GPU с 4GB памяти: tile=200
  - CPU режим: тайлинг отключен
- Для ручной настройки укажите параметр `tile` явно (это отключит автоопределение)
- При нехватке памяти GPU уменьшите размер тайла или используйте `fp32=false`

## Безопасность

⚠️ **Важно:** По умолчанию CORS разрешен для всех источников (`allow_origins=["*"]`).
Для продакшена рекомендуется настроить конкретные домены в `api_server.py`.

## Troubleshooting

### Ошибка "CUDA out of memory"
Уменьшите размер тайла или используйте меньшую модель:
```bash
curl -X POST "http://localhost:8000/upscale" \
  -F "file=@input.jpg" \
  -F "tile=400"
```

### Модель не загружается
Модели автоматически скачиваются при первом использовании. Убедитесь, что есть доступ к интернету и достаточно места на диске.

### Медленная обработка
- Используйте GPU вместо CPU
- Убедитесь, что `fp32=false` (используется fp16 по умолчанию)
- Используйте более легкие модели для быстрой обработки


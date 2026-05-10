# Retail Analytics (UI-Controlled Multi-Camera Shelf Analytics)

Bu proje; birden fazla kameradan (USB webcam / RTSP / video dosyasi) ayni anda goruntu alip **raf bazli (shelf)** ziyaret ve bekleme suresi (dwell) analitigi ureten, her seyi **tek bir Streamlit web arayuzunden** yonetebileceginiz bir retail analytics sistemidir.

Temel mantik:

- **Her kamera = tek bir raf (shelf)** (polygon/zone yok)
- Bir kisi kamerada gorunur oldugu surece **o rafla etkileisimde** kabul edilir
- YOLOv8 ile insan tespiti, Deep SORT ile takip (visitor_id)
- Age/Gender modeli varsa (UTKFace ile egitilmis), demografi tahmini yapilir
- Tum kamera verisi tek bir **SQLite** veritabaninda toplanir

---

## 1) Mimari (High-Level)

- `CameraManager`: Konfigdeki kameralari yukler, UI uzerinden kamera worker start/stop/restart/test yonetir
- `CameraWorker`: Her kamera icin arkaplanda calisan thread (capture + analytics + preview)
- `AnalyticsEngine`: Detection + tracking + inference + dwell + DB eventleri (kamera basina stateful)
- `EventAggregator`: SQLite yazimlarini tek thread uzerinden serialize eder (multi-camera icin stabil)
- `StreamManager`: UI icin her kameranin en guncel annotated frame’ini RAM’de tutar
- `DashboardService`: UI icin DB’den istatistik/ziyaret verilerini okur

---

## 2) Klasor Yapisi

```
retail_analytics/
  app/
    api/                      # (Opsiyonel) FastAPI - mevcut ama UI zorunlu degil
    database/
      models.py               # shelf_visits tablosu
      repository.py           # DB CRUD + istatistik
    model/
      dataset.py              # UTKFace dataset loader
      train.py                # age/gender egitim
      inference.py            # age/gender inference
    services/
      dashboard_service.py    # UI icin DB okuma
    ui/
      control_panel.py        # Streamlit UI (Dashboard/Cameras/Analytics/Settings)
    vision/
      analytics_engine.py     # kamera-bazli analytics core
      camera_manager.py       # UI-controlled camera lifecycle
      camera_worker.py        # background worker (per camera)
      detector.py             # YOLOv8 wrapper
      tracker.py              # Deep SORT wrapper
      event_aggregator.py     # tek DB writer queue
      stream_manager.py       # latest frames for UI
  config/
    config.yaml               # cameras + pipeline ayarlari
  data/
    UTKFace/                  # UTKFace resimleri (part1/part2/... olabilir)
    analytics.db              # SQLite (olusturulur)
  models/
    age_gender_model.pth      # (egitince olusur / buraya kopyalanir)
  logs/
  run_ui.py                   # UI launcher (streamlit komutu yazmadan)
  start_ui.bat                # Windows icin double-click launcher
  requirements.txt
```

---

## 3) Gereksinimler

- Windows 10/11 (bu repo Windows icin optimize)
- Python **3.10+** (onerilen: 3.11)
- Kamera driver’lari (USB webcam) / RTSP erisimi (network)

---

## 4) Kurulum (Windows - PowerShell)

Proje klasorune girin:

```powershell
cd C:\Users\kajek24\Desktop\retail_analytics_project\retail_analytics
```

Sanal ortam:

```powershell
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Bagimliliklar:

```powershell
pip install -r requirements.txt
```

Notlar:

- `ultralytics` ilk calismada YOLO agirligini indirmeye calisabilir. Repo kokunde `yolov8n.pt` varsa download gerekmez.
- GPU kullanmak isterseniz `torch` icin CUDA wheel kurulumunu PyTorch resmi sitesinden secmelisiniz.

---

## 5) Kamera Konfigurasyonu (YAML)

Kameralari `config/config.yaml` icindeki `cameras:` listesine ekleyin.

Ornek:

```yaml
cameras:
  - id: "cam_1"
    shelf_name: "Snacks"
    source: 0

  - id: "cam_2"
    shelf_name: "Drinks"
    source: "rtsp://user:pass@ip/stream"

  - id: "cam_3"
    shelf_name: "Electronics"
    source: "data/video.mp4"
```

`source` degerleri:

- `0`, `1`, `2` ... : USB webcam index
- `rtsp://...` : RTSP
- `data/video.mp4` : dosya

---

## 6) UI’yi Baslatma (Terminal Komutsuz Streamlit)

En kolay yol:

- `start_ui.bat` dosyasina **double-click**

veya PowerShell ile:

```powershell
.\venv\Scripts\python.exe run_ui.py
```

UI adresi:

- `http://localhost:8501`

---

## 7) UI Kullanimi (Adim Adim)

### Dashboard

- Toplam ziyaretci, aktif ziyaretci, ortalama dwell
- Raf populerligi, aktif ziyaretci sayisi (Plotly)
- Gender/Age dagilimi (Plotly)

Filtreler:

- Camera / Shelf
- Date + Time range

### Cameras (Kamera Yonetimi)

Her kamera kartinda:

- Camera ID, Shelf name, Source
- Status: `OFFLINE / RUNNING / ERROR / STOPPED`
- Thumbnail preview + FPS + aktif kisi sayisi

Butonlar:

- **Start**: analytics + DB yazimi baslar
- **Stop**: worker durur, acik kalmis visit’ler kapanir
- **Restart**: stop + start
- **Test (10s)**: DB’ye yazmadan 10 saniye preview (otomatik durur)
- **Open Camera**: UI icinde live preview panelini acar

### Analytics

- Filtreleyerek ziyaret kayitlarini tablo olarak goruntuleme

### Settings

- DB path gosterir
- `config/config.yaml` icerigini gosterir
- Reload config / Stop all / Shutdown runtime

---

## 8) Age/Gender Model (Opsiyonel)

Egitim icin UTKFace resimlerini su dizine koyun (alt klasorler serbest):

- `data/UTKFace/part1/*.jpg`
- `data/UTKFace/part2/*.jpg`
- `data/UTKFace/part3/*.jpg`

UTKFace dosya adi formati:

```text
<age>_<gender>_<race>_<timestamp>.jpg
```

Ornek: `92_1_2_20170110175823500.jpg`

- `92`: gercek yas
- `1`: Female (`0` = Male, `1` = Female)
- `2`: race etiketi; bu projede egitim icin kullanilmiyor
- Yeni yas araliklariyla bu ornek `85+` grubuna girer

Bu projede daha ince yas araliklari kullanilir:

```yaml
age_gender:
  input_size: 96
  face_detector: "haar"
  min_votes: 5
  vote_min_conf: 0.60
  age_bins: [13, 18, 25, 35, 45, 55, 65, 75, 85]
  age_labels: ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
```

Not: `age_bins` esik degerdir. Ornegin `13` degeri, `age < 13` icin `0-12` grubunu uretir.

Egitim (opsiyonel, UI’dan bagimsiz):

```powershell
.\venv\Scripts\python.exe main.py train --data_dir data/UTKFace --save_dir models --epochs 30 --batch_size 32 --input_size 96 --device cpu
```

GPU varsa daha hizli egitim icin:

```powershell
.\venv\Scripts\python.exe main.py train --data_dir data/UTKFace --save_dir models --epochs 30 --batch_size 64 --input_size 96 --device cuda --num_workers 2
```

Model dosyasi:

- Varsayilan: `models/age_gender_model.pth`
- Yol ayari: `config/config.yaml` → `age_gender.model_path`
- Yas araliklari: `config/config.yaml` → `age_gender.age_bins` + `age_gender.age_labels` (**degistirirseniz modeli yeniden egitin**)

**Modeli yeniden egitmeniz gerekiyor.** Cunku yas araliklari degistiginde modelin `age_head` cikis sinifi sayisi da degisir. Eski 4 sinifli model (`0-18`, `18-30`, `30-50`, `50+`) yeni 10 sinifli araliklarla dogru calismaz.

Cinsiyet/yas dogrulugunu artirmak icin bu repo su ayarlari kullanir:

- Egitimde train ve validation transformlari ayridir; validation setine augmentation uygulanmaz.
- Egitimde gender ve age siniflari icin balanced `CrossEntropyLoss` kullanilir.
- `input_size: 96` daha fazla yuz detayi verir.
- Canli calismada `face_detector: "haar"` kisi crop'u icindeki yuzu netlestirir.
- `min_votes: 5` ve `vote_min_conf: 0.60` tek karelik hatali tahminleri azaltir.

Model yoksa sistem yine calisir; gender/age `Unknown` kalir.

---

## 9) Veritabani (SQLite)

DB dosyasi:

- `data/analytics.db`

Tablo:

- `shelf_visits`

Alanlar:

- `camera_id`, `shelf_name`
- `visitor_id` (Deep SORT track id; kamera icinde benzersiz)
- `gender`, `age_group`
- `enter_time`, `exit_time`, `duration_seconds`

Sifirlamak icin:

- `data/analytics.db` dosyasini silin (UI tekrar olusturur)

---

## 10) Analytics NasIl Calisiyor?

Her `CameraWorker`:

1) Kaynagi acar (webcam/rtsp/video)
2) Her frame icin `AnalyticsEngine.process_frame()` cagirir:
   - YOLOv8 ile person detect
   - Deep SORT ile tracking (visitor_id)
   - (Opsiyonel) head ROI icinde **face detector (haar)** ile daha iyi age/gender crop
   - Kisi ilk gorununce `open_visit`, kaybolunca `close_visit`
3) Annotated frame `StreamManager`’a yazilir (UI thumbnail/live)

DB yazimlari `EventAggregator` thread’i ile tek noktadan yapilir (multi-camera icin stabil).

---

## 11) Troubleshooting

### Kamera OFFLINE / acilmiyor

- `config/config.yaml` icinde `source: 0` dogru mu? (USB webcam genelde 0)
- UI > Cameras > **Test (10s)** ile deneyin
- Bazi Windows webcam driver’lari capture resolution set etmeye takilabilir:
  - `config/config.yaml` icinde `capture.set_capture_resolution: false` kalsin

### YOLO agirlik indirimi / dosya bulunamadi

- Internet yoksa `yolov8n.pt` dosyasini proje kokune koyun

### Performans dusuk (FPS)

- `yolov8n.pt` kullanin (n = en hizli)
- Kamera sayisini azaltin / cozumurlugu dusurun (`capture.width/height`)
- `age_gender.infer_interval` degerini artirin (or: 30)

### Streamlit sayfasi donuyor / gec geliyor

- Live preview refresh interval’i yuksek yapin (0.5s–1.0s)
- Cok fazla kamerayi ayni anda RUNNING yapmayin (CPU)

---

## 12) Notlar

- Bu repo icinde `main.py`/FastAPI mevcut olabilir; ancak **gunluk kullanim icin tek gerekli giris**: `run_ui.py` (veya `start_ui.bat`).

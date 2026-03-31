# VTOL Transfer Öğrenimi — Proje README

## Proje Özeti

PPO algoritmasının basit ortamdan karmaşık ortama transfer edilebilirliğini araştıran
simülasyon tabanlı bir pekiştirmeli öğrenme çalışması.

## Araştırma Hipotezi

> Basit ortamda (gürültüsüz 4 waypoint) ön eğitim yapılmış bir PPO politikasını
> karmaşık ortamda (gürültülü 4 waypoint) ince ayarlamak; örnek verimliliği ve
> nihai performans açısından sıfırdan eğitimden üstündür.

---

## Ortamlar

### Basit Ortam — `VTOLEnv(complex_mode=False)`
- 4 rastgele waypoint, sırayla ziyaret et
- İdeal PVTOL dinamiği (gürültü yok, hava direnci yok)
- Waypoint arası minimum mesafe: 4 metre

### Karmaşık Ortam — `VTOLEnv(complex_mode=True)`
- Aynı görev, aynı waypoint yapısı
- Hava direnci aktif (rho=1.225, Cd=0.1)
- Türbülans: N(0, 0.4) Gaussian gürültü

### Gözlem Uzayı (8 boyut)
```
[x, y, θ, vx, vy, ω, dx, dy]
```
- `dx, dy` → aktif waypoint'e uzaklık

### Aksiyon Uzayı (2 boyut)
```
[thrust, tork]  ∈  [-1, 1]
```
- thrust → [0, 20] N'a eşlenir
- tork   → [-1, 1] Nm

### Ödül Fonksiyonu
```
r = -1.0 × mesafe
  - 0.05 × |θ|
  - 0.01 × (vx² + vy²)
  + 20.0  (waypoint'e ulaşıldıysa)
```

---

## 3 Strateji

| Strateji | Açıklama | Toplam Adım |
|----------|----------|-------------|
| S1 Zero-shot | Basit'te N adım → Karmaşık'ta direkt test | N |
| S2 Transfer  | Basit'te N adım → Karmaşık'ta K adım ince ayar | N + K |
| S3 Sıfırdan  | Karmaşık'ta N+K adım | N + K |

> S2 ve S3 toplam bütçe eşittir → karşılaştırma adildir.

---

## Parametreler

```python
N_SIMPLE  = 1_000_000   # Basit ortam ön eğitim adımı
K_COMPLEX =   500_000   # Karmaşık ortam ince ayar adımı
N_ENVS    = 8           # Paralel ortam sayısı
EVAL_FREQ = 20_000      # Değerlendirme sıklığı
N_EVAL    = 200         # Değerlendirme episode sayısı

# PPO
learning_rate = 3e-4
n_steps       = 2048
batch_size    = 512
n_epochs      = 10
gamma         = 0.99
device        = "cpu"

# S2 ince ayar learning rate (catastrophic forgetting önlemi)
ft_learning_rate = 1e-4
```

---

## Metrikler

- **Ortalama ödül** — yüksek olması iyi (negatif değerler)
- **Waypoint başarı oranı (%)** — 4 waypoint'ten kaçı tamamlandı
- **Öğrenme eğrisi** — kaçıncı adımda ne kadar iyi

---

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `vtol_transfer_v2.py` | Ana kod |
| `simple_pretrained.zip` | Basit ortamda eğitilmiş model |
| `transfer_finetuned.zip` | Transfer + ince ayar modeli |
| `scratch_complex.zip` | Sıfırdan karmaşık ortamda eğitilmiş model |
| `vtol_transfer_v2_results.png` | Sonuç grafiği |

---

## Kurulum

```bash
pip install gymnasium stable-baselines3 matplotlib
```

## Çalıştırma

```bash
python vtol_transfer_v2.py
```

---

## Sonraki Adımlar

- [ ] Çoklu seed (3-5 farklı seed ile tekrar et, ortalama al)
- [ ] Domain randomization (kütle, sürtünme, motor gürültüsü)
- [ ] Gazebo SIL entegrasyonu (ROS2 Humble + Gazebo Harmonic)
- [ ] Uçuş yolu görselleştirme (waypoint izleri)
- [ ] Transition modu araştırması (airspeed sensörü eklendikten sonra)

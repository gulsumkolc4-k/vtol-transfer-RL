# Araştırma Notları — VTOL Transfer Öğrenimi

---

## Genel Bağlam

**Tez başlığı:**
"Pekiştirmeli Öğrenme Modellerinin Transferinde Genellenebilirliğin Korunması"

**Danışman notu (özet):**
Basit model + karmaşık model kurgusu önerildi.
Basit ortamda N adım, karmaşık ortamda K adım ince ayar, karmaşık ortamda N+K adım
sıfırdan — bu 3 stratejiyi karşılaştır.

---

## Kararlar ve Gerekçeleri

### Görev seçimi
**Karar:** Hover yerine 4 waypoint yörünge takibi kullanıldı.

**Gerekçe:**
Hover statik bir görev, ajan hareket etmeyi öğrenmiyor.
Waypoint navigasyonunda görev yapısı aynı kalırken fizik zorlaşıyor.
Transfer burada gerçekten anlamlı iş yapıyor.
Jüri itirazına ("iki ortam zaten benzer") karşı güçlü argüman.

---

### Basit / Karmaşık ortam ayrımı
**Karar:** Tek sınıf (`VTOLEnv`), `complex_mode` parametresi ile ayrım.

**Gerekçe:**
- Görev aynı (4 waypoint), sadece fizik değişiyor
- Bu izole bir değişken → transfer etkisi net ölçülüyor
- `HoverEnv` tamamen kaldırıldı

**Basit:** İdeal PVTOL dinamiği, gürültü yok
**Karmaşık:** Hava direnci + türbülans (N(0, 0.4))

---

### Eşit bütçe garantisi
**Karar:** S3 toplam adımı = N_SIMPLE + K_COMPLEX

**Gerekçe:**
S2 ve S3 farklı adım sayısıyla karşılaştırılsaydı
"S2 daha fazla eğitildi" itirazı geçerli olurdu.
Karşılaştırma ancak eşit bütçeyle adil olur.

---

### S2 ince ayar learning rate
**İlk deneme:** `5e-5` → çok küçük, politika güncellenemedi
**Düzeltme:** `1e-4`

**Grafikteki belirti:**
Öğrenme eğrisinde S2 (mavi) hiç hareket etmiyordu,
S1 zero-shot çizgisiyle aynı seviyede kaldı.

---

### Adım sayıları
**İlk deneme:** N=300K, K=200K → yetersiz, WP başarısı %2'nin altında
**Düzeltme:** N=1M, K=500K

**Gerekçe:**
PPO'nun 4 waypoint görevini öğrenmesi için
1-2M adım gerekiyor. 300K çok erken.

---

## Sonuçlar (v1 — 300K + 200K)

| Strateji | Ödül | WP Başarı |
|----------|------|-----------|
| S1 Zero-shot | -356.8 | %2.0 |
| S2 Transfer  | -350.9 | %1.6 |
| S3 Sıfırdan  | -358.5 | %1.1 |

**Hipotez:** Desteklendi (S2 > S3)
**Sorun:** Fark çok küçük, adım sayısı yetersiz

**Öğrenme eğrisi gözlemi:**
- S3 (yeşil) -900'den -350'ye çıktı → öğreniyor
- S2 (mavi) hiç hareket etmedi → lr çok küçüktü
- Waypoint başarısı her stratejide %2'nin altında → adım sayısı yetersiz

---

## Sonraki Deney (v2 — 1M + 500K)

Beklenti:
- WP başarısı %20-40'a çıkmalı
- S2 öğrenme eğrisi S3'ten daha erken yükselmeli
- Fark daha belirgin olmalı

---

## Gazebo / Donanım Yol Haritası

```
Katman 1 (mevcut): Pure Python sim — PPO eğitimi
     ↓
Katman 2 (hedef) : Gazebo SIL — ROS2 Humble + Gazebo Harmonic
     ↓
Katman 3 (bonus) : HIL — Pixhawk + companion computer
     ↓
Katman 4 (bonus) : Gerçek uçuş
```

**Gazebo notları:**
- Sistemde: ROS2 Humble + Gazebo Harmonic
- VTOL modeli Gazebo Classic'ten Harmonic'e taşınıyor (devam ediyor)
- Airspeed sensörü simde yok → hazır model include edilecek
- Gerçek uçuş kaydı yok → tez simülasyon tabanlı, bu normal

---

## Transition Araştırması (ileride)

**Plan:**
Airspeed hazır model include edildikten sonra
drone modu (basit) → transition (karmaşık) denenecek.

**Neden güçlü argüman:**
- VTOL'un özü bu geçiş
- "Airspeed'siz RL transition" özgün katkı olur
- Bu uçağın ilk sim transition denemesi olabilir

**Risk:**
Transition sim'de çalışmayabilir.
Bu durumda negatif sonuç da bulgudur, felaket değil.
Yörünge transferi tek başına yeterli ve savunulabilir.

---

## Kullanılan Kaynaklar

- Schulman et al. 2017 — PPO makalesi
- Sutton & Barto — RL: An Introduction
- Zhu et al. 2023 — Transfer Learning in Deep RL: A Survey
- Bengio et al. 2009 — Curriculum Learning
- Peng et al. 2018 — Sim-to-Real Transfer with Dynamics Randomization
- Stable Baselines3 dokümantasyonu
- Gymnasium dokümantasyonu

---

## Açık Sorular

- [ ] Çoklu seed deneyi yapılmadı — sonuçlar şansa bağlı olabilir
- [ ] Domain randomization henüz yok — sim-to-real geçişte sorun çıkabilir
- [ ] S2 ince ayar lr optimumu bulunmadı — 1e-4 deneniyor
- [ ] Waypoint sayısı (4) optimum mu? Daha azla başlamak daha iyi olabilir

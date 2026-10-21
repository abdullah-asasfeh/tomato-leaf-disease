# Produce Freshness: Dataset Comparison

Planning notes for the next project: a phone camera app you point at a tomato (or other
produce) in a store, and it says whether it's fresh, getting old, or rotten.

This folder is separate from the leaf-disease app on purpose; nothing here touches
`streamlit_app.py` or `best_field_model.pt`.

_Researched October 2026. Licenses and contents can change, so re-check each page before downloading._

## Comparison

| Dataset | Produce | Images | Labels | Photo conditions | License | Verdict |
|---|---|---|---|---|---|---|
| [AgriFreshNET](https://data.mendeley.com/datasets/42m5tb7yv9) (Jul 2026) | 8 kinds × 3 stages = 24 classes, **incl. tomato** | 14,500 (~14,100 after de-duplication, per users) | Fresh / Semi-fresh / Rotten + estimated shelf life (days) | Phone photos in markets, supermarkets, homes and farms; mixed lighting and backgrounds | CC BY 4.0 ✅ | **Best match for a store app.** Has tomato at all three stages (see note below). Download is ~329 MB (`Raw Data.zip` + `Processed_Data.zip`). |
| [Khatun tomato set](https://data.mendeley.com/datasets/s42kpg8h37/1) (2023) | Tomato | 2,986 original (+10,000 augmented) | Fresh / Rotten, plus Mature / Immature | Real photos from a farm university (Bangladesh) | CC BY 4.0 ✅ | **Best tomato-only set.** Cleaner copy on [Hugging Face](https://huggingface.co/datasets/Project-AgML/tomato_quality_classification): 1,350 fresh, 636 rotten. |
| [Tomato detection set](https://data.mendeley.com/datasets/vy5y8p2kdb/1) (May 2026) | Tomato | 1,600 (800 original) | Fresh / Rotten, YOLOv8 boxes | "Different lighting and backgrounds" | CC BY 4.0 ✅ | Small extra tomato data. |
| [Fresh and Stale Images](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) (Kaggle, 2021) | 6 kinds incl. tomato | ~14k (≈20 augmented copies per photo) | Fresh / Stale | Phone photos | CC0 on Kaggle ✅ | Fine for practice. 1.5 GB, mostly augmented copies. |
| [Fruit & Vegetable Disease](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) (Kaggle, 2024) | 14 kinds | not stated (5.1 GB) | Healthy / Rotten | Collected from "public sources" (likely web images) | CC0 on Kaggle ⚠️ | Wide coverage, but the uploader collected the images, so CC0 may not hold for every image. |
| [FruitNet](https://data.mendeley.com/datasets/b6fftwbr2v/3) | 6 Indian fruits, **no tomato** | 14,700+ | Good / Bad / Mixed | Plain backgrounds | ❓ not stated | Skip for tomatoes. |
| [FreshLab Tomato](https://ieee-dataport.org/documents/freshlab-tomato-longitudinal-shelf-life-dataset) (Sep 2026) | Tomato | 1,080 | Spoilage score 1–5 | Same 12 tomatoes photographed over 10 days | not stated | Paid IEEE subscription. Skip. |

## Recommendation

1. **Start with AgriFreshNET.** It's the only large set photographed in real stores, which is
   where the app will be used. That's our best defence against the drop we saw with the
   leaf model (99.3% on lab photos, 62% on field photos).
2. **Add the Khatun tomato set** for more tomato examples.
3. **AgriFreshNET tomato check:** the official page doesn't list the classes, but two independent
   projects that use it list the 8 kinds as banana, bitter melon, cucumber, eggplant, orange,
   papaya, pineapple and **tomato**, each at Fresh / Semi-fresh / Rotten. Final confirmation and
   per-class counts come when we download it (v2: adds the Semi-fresh class).
4. **Take our own store photos:** 100–200 per class from a local store, used **only for testing**,
   never for training. That's the honest measure of real-world accuracy.

## Things to watch

- **Augmented copies:** several sets pad their counts with flipped/rotated/zoomed copies of the
  same photo. Split train/test on the **original** photos only, or near-duplicates leak into the
  test set and accuracy looks better than it is.
- **Licenses:** CC BY 4.0 allows any use, including in an app, as long as we credit the authors
  (in the app and the README).
- **Don't commit the data:** `.gitignore` already excludes `data/`, `dataset/`, `datasets/` and
  archives. Keep downloads there.
- **What a photo can't show:** internal damage, bruises underneath, or taste. The app should
  say so, like the leaf app's "Limitations" tab.

## Decisions (from the PR #3 review)

- **v1:** tomato only, Fresh / Rotten, trained on the Khatun set, Streamlit app with the phone camera.
- **v2:** add AgriFreshNET's tomato images and the Semi-fresh class, then 1–2 more kinds of produce.
- **Native app:** only once our own store photos show the accuracy is good enough.

Work is tracked in the [v1 milestone](https://github.com/abdullah-asasfeh/tomato-leaf-disease/milestone/1).

# Produce Freshness: Dataset Comparison

Planning notes for the next project: a phone camera app you point at a tomato (or other
produce) in a store, and it says whether it's fresh, getting old, or rotten.

This folder is separate from the leaf-disease app on purpose; nothing here touches
`streamlit_app.py` or `best_field_model.pt`.

_Researched October 2026. Licenses and contents can change, so re-check each page before downloading._

## Comparison

| Dataset | Produce | Images | Labels | Photo conditions | License | Verdict |
|---|---|---|---|---|---|---|
| [AgriFreshNET](https://data.mendeley.com/datasets/42m5tb7yv9) (Jul 2026) | 24 kinds | 14,500 | Fresh / Semi-fresh / Rotten + estimated shelf life (days) | Phone photos in markets, supermarkets, homes and farms; mixed lighting and backgrounds | CC BY 4.0 ✅ | **Best match for a store app.** Has a "semi-fresh" level. ⚠️ Not yet confirmed that tomato is one of the 24, and the file list was empty when checked. |
| [Khatun tomato set](https://data.mendeley.com/datasets/s42kpg8h37/1) (2023) | Tomato | 2,986 original (+10,000 augmented) | Fresh / Rotten, plus Mature / Immature | Real photos from a farm university (Bangladesh) | CC BY 4.0 ✅ | **Best tomato-only set.** Cleaner copy on [Hugging Face](https://huggingface.co/datasets/Project-AgML/tomato_quality_classification): 1,350 fresh, 636 rotten. |
| [Tomato detection set](https://data.mendeley.com/datasets/vy5y8p2kdb/1) (May 2026) | Tomato | 1,600 (800 original) | Fresh / Rotten, YOLOv8 boxes | "Different lighting and backgrounds" | CC BY 4.0 ✅ | Small extra tomato data. |
| [Fresh and Stale Images](https://gts.ai/dataset-download/fresh-and-stale-images-of-fruits-and-vegetables) (Kaggle) | 6 kinds incl. tomato | ~14k | Fresh / Stale | Phone photos, heavily augmented | ❓ unconfirmed | Fine for practice; check the license before public use. |
| [Fruit & Vegetable Disease](https://gts.ai/dataset-download/fruit-and-vegetable-disease-dataset/) (Kaggle) | 14 kinds | not stated | Healthy / Rotten | Collected from "public sources" (likely web images) | ❓ unconfirmed | Wide coverage, unclear image rights. |
| [FruitNet](https://data.mendeley.com/datasets/b6fftwbr2v/3) | 6 Indian fruits, **no tomato** | 14,700+ | Good / Bad / Mixed | Plain backgrounds | ❓ not stated | Skip for tomatoes. |
| [FreshLab Tomato](https://ieee-dataport.org/documents/freshlab-tomato-longitudinal-shelf-life-dataset) (Sep 2026) | Tomato | 1,080 | Spoilage score 1–5 | Same 12 tomatoes photographed over 10 days | not stated | Paid IEEE subscription. Skip. |

## Recommendation

1. **Start with AgriFreshNET.** It's the only large set photographed in real stores, which is
   where the app will be used. That's our best defence against the drop we saw with the
   leaf model (99.3% on lab photos, 62% on field photos).
2. **Add the Khatun tomato set** for more tomato examples.
3. **Before committing:** confirm on the AgriFreshNET page that tomato is one of the 24 classes
   and that the files download.
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

## Open questions

- Which produce to start with: tomato only, or tomato + 1–2 others?
- Classes: Fresh / Rotten, or Fresh / Semi-fresh / Rotten?
- First version as the existing Streamlit app with the phone camera, or straight to a native app?

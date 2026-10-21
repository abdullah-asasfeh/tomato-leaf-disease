# 🍅 Tomato Leaf Disease Classifier

مشخّص أمراض أوراق البندورة — صوّر ورقة بندورة واحدة، والنموذج بيعطيك التشخيص المرجّح ويوضّح وين نظر بالصورة.

A Streamlit app that classifies a photo of a single tomato leaf into one of 10 classes
(9 diseases + healthy) and shows a Grad-CAM heatmap of where the model looked.

## Model

| | |
|---|---|
| Architecture | EfficientNet-B0 (ImageNet-pretrained, fine-tuned) |
| Data | PlantVillage (lab photos) + PlantDoc (field photos) |
| Explainability | Grad-CAM |
| Weights | `best_field_model.pt` |

| | Lab photos | Field photos |
|---|---|---|
| Before field fine-tuning | 99.3% | 30.4% |
| After field fine-tuning | 99.3% | 62.0% |

The app shows "not sure" instead of a diagnosis when confidence is below 60% or the top two
classes are within 20% of each other.

**Classes:** Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot,
Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy.

## Run locally

Requires Python 3.10–3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501. PyTorch is installed from the CPU-only index, so no GPU is needed.

## Project layout

```
streamlit_app.py       the app (UI, model loading, prediction, Grad-CAM)
best_field_model.pt    trained model weights
requirements.txt       pinned dependencies
.streamlit/config.toml theme
```

## Limitations

- It only knows these 10 tomato-leaf classes; any other image is forced into one of them.
- Field accuracy (~62%) is measured on a small test set (79 images), so treat it as approximate.
- Spider mites had no field photos in training.

⚠️ Experimental model for learning purposes. Not a substitute for an agronomist's diagnosis.

## Contributing

1. `git pull` on `main` before starting.
2. Work on a branch: `abdullah/<topic>` or `salem/<topic>`.
3. Open a Pull Request; the other person reviews and merges.

Developed by Abdullah Asasfeh.

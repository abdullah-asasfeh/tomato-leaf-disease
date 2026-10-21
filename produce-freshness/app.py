"""
فاحص طزاجة البندورة — Tomato freshness checker (v1: fresh / rotten) — Streamlit

Point the phone camera at one tomato; the model says fresh, rotten, or "not sure",
and shows where it looked (Grad-CAM). Settings come from config.py.
Until fresh_model.pt is trained (#11), an untrained placeholder model is used and
the app says so.

    streamlit run produce-freshness/app.py
"""
import threading

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageOps, UnidentifiedImageError

import config

st.set_page_config(page_title="فاحص طزاجة البندورة", page_icon="🍅",
                   layout="centered", initial_sidebar_state="collapsed")

GITHUB_URL = "https://github.com/abdullah-asasfeh/tomato-leaf-disease"
MIN_SIDE = 64          # smaller photos can't be judged
DISPLAY_SIDE = 640     # long side of the images shown on the page

# ---------------------------------------------------------------
# 0) التصميم
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

/* The result card carries its own light surface and ink, so it reads the same under
   Streamlit's light or dark theme (the page can't tell which one is active), and stays
   high-contrast under bright store lighting. */
:root {
  --ink: #1c1917;
  --fresh: #2e7d32;  --fresh-tint: #eef6ee;
  --rotten: #9a3412; --rotten-tint: #fbf0ea;
  --unsure: #57534e; --unsure-tint: #f3f2f0;
  --line: rgba(120,113,108,.35);
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
}

html, body, .stApp { font-family: 'Tajawal', sans-serif; }
.stApp, .stMarkdown, p, h1, h2, h3, h4, label, li { direction: rtl; text-align: right; }
::selection { background: rgba(46,125,50,.22); }
.stApp { accent-color: var(--fresh); }

/* Phone: no tap flash, instant taps, controls don't select on long-press */
html { -webkit-tap-highlight-color: transparent; -webkit-text-size-adjust: 100%; }
button, [role="button"], [role="tab"], label { touch-action: manipulation; }
button, [role="button"], [role="tab"] { user-select: none; -webkit-user-select: none; }
button:active { transform: scale(0.97); transition: transform 120ms var(--ease-out); }

.head h1 { font-size: 1.75rem; font-weight: 800; margin: 0 0 .25rem; line-height: 1.25; }
.head p  { margin: 0 0 1rem; opacity: .8; font-size: 1.05rem; }

.notice { border: 1px solid var(--line); border-radius: 12px; padding: .7rem 1rem;
          margin: .25rem 0 1rem; font-size: .95rem; background: var(--unsure-tint); color: var(--ink); }

/* Result */
.verdict { border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.25rem 1.2rem;
           margin: 1rem 0 .75rem; color: var(--ink);
           box-shadow: 0 1px 2px rgba(28,25,23,.06), 0 4px 14px rgba(28,25,23,.06); }
.verdict.fresh  { background: var(--fresh-tint);  border-color: var(--fresh); }
.verdict.rotten { background: var(--rotten-tint); border-color: var(--rotten); }
.verdict.unsure { background: var(--unsure-tint); }
.verdict h2 { margin: 0; font-size: 2.25rem; font-weight: 800; line-height: 1.15; }
.verdict.fresh h2  { color: var(--fresh); }
.verdict.rotten h2 { color: var(--rotten); }
.verdict.unsure h2 { color: var(--unsure); }
.verdict p { margin: .35rem 0 0; font-size: 1.05rem; }
.verdict { transition: opacity 220ms var(--ease-out), transform 220ms var(--ease-out); }
@starting-style { .verdict { opacity: 0; transform: translateY(8px); } }

/* Freshness scale: fresh (start) -> not sure -> rotten (end); marker = chance of rot */
.scale { margin: 1.1rem 0 .2rem; }
.scale-track { position: relative; height: 12px; border-radius: 999px; display: flex;
               overflow: visible; direction: rtl; }
.scale-track span { height: 100%; }
.z-fresh  { background: var(--fresh);  opacity: .85; border-radius: 0 999px 999px 0; }
.z-unsure { background: repeating-linear-gradient(135deg, var(--line) 0 4px, transparent 4px 8px);
            border-block: 1px solid var(--line); box-sizing: border-box; }
.z-rotten { background: var(--rotten); opacity: .85; border-radius: 999px 0 0 999px; }
.marker { position: absolute; top: 50%; width: 22px; height: 22px; border-radius: 50%;
          background: #fff; border: 3px solid var(--ink);
          box-shadow: 0 1px 4px rgba(0,0,0,.25);
          transform: translate(50%, -50%);
          transition: inset-inline-start 300ms var(--ease-out); }
@starting-style { .marker { inset-inline-start: 50% !important; } }
.scale-labels { display: flex; justify-content: space-between; font-size: .9rem;
                margin-top: .45rem; opacity: .85; }
.scale-readout { font-size: .95rem; margin-top: .5rem; font-variant-numeric: tabular-nums; }

@media (prefers-reduced-motion: reduce) {
  .verdict, .marker, button:active { transition: opacity 150ms ease; transform: none; }
  @starting-style { .verdict { transform: none; } }
}
@media (hover: hover) and (pointer: fine) {
  .foot a:hover { text-decoration-thickness: 2px; }
}

.foot { font-size: .85rem; opacity: .75; text-align: center !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# 1) النموذج (مرة وحدة بس)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    """Returns (model, is_placeholder). Before fresh_model.pt exists, an untrained
    model with fixed weights stands in so the screens can be tested."""
    torch.manual_seed(config.SEED)
    m = config.build_model(pretrained=False)
    placeholder = not config.MODEL_PATH.is_file()
    if not placeholder:
        m.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
    m.eval()
    return m, placeholder


model, is_placeholder = load_model()
tf = config.eval_transform()
# Grad-CAM runs a backward pass on the shared model, so two visitors at once must
# take turns, or one would get the other's heatmap.
model_lock = threading.Lock()


def analyze(img):
    """Probabilities, predicted class, the photo at display size, and a Grad-CAM
    overlay (None when no region stood out)."""
    saved = {}

    def hook(module, inp, out):
        saved["act"] = out
        out.register_hook(lambda g: saved.__setitem__("grad", g))

    with model_lock:
        handle = model.features[-1].register_forward_hook(hook)
        try:
            with torch.enable_grad():
                out = model(tf(img).unsqueeze(0))
                probs = F.softmax(out, dim=1)[0].detach().numpy()
                cls = int(probs.argmax())
                model.zero_grad()
                out[0, cls].backward()
        finally:
            handle.remove()
        weights = saved["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * saved["act"]).sum(dim=1))[0].detach()

    # The model sees the photo squashed to a square, so stretching the map back to
    # the photo's own shape puts each spot where it belongs.
    shown = img.copy()
    shown.thumbnail((DISPLAY_SIDE, DISPLAY_SIDE))
    if float(cam.max()) <= 1e-6:
        return probs, cls, shown, None   # nothing stood out: no heatmap to show
    cam = cam / cam.max()
    cam = F.interpolate(cam[None, None], size=(shown.height, shown.width),
                        mode="bilinear")[0, 0].numpy()
    heat = (colormaps["jet"](cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(shown, Image.fromarray(heat), alpha=0.45)
    return probs, cls, shown, overlay


def open_photo(uploaded):
    """Upright RGB photo, or (None, error message)."""
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "ما قدرنا نفتح هالصورة. جرّب صورة JPG أو PNG أو WEBP."
    if min(img.size) < MIN_SIDE:
        return None, f"الصورة صغيرة كتير ({img.width}×{img.height}). صوّر الحبة من أقرب."
    return img, None


VERDICTS = {
    "fresh":  ("طازجة", "ما بيّن على الحبة علامات تلف واضحة."),
    "rotten": ("تالفة", "النموذج شاف علامات تلف، مثل بقع داكنة أو عفن أو تجاعيد."),
    "unsure": ("غير متأكد", "الصورة مش كافية للحكم. جرّب حبة وحدة بإضاءة أوضح ومن أقرب."),
}


def verdict_html(probs, cls):
    top2 = np.sort(probs)[-2:]
    unsure = top2[1] < config.THRESHOLD or top2[1] - top2[0] < config.MARGIN
    key = "unsure" if unsure else config.CLASSES[cls]
    title, text = VERDICTS[key]

    rot = float(probs[config.CLASSES.index("rotten")])
    # Not-sure band: where neither class reaches THRESHOLD with a MARGIN lead.
    lo = 1 - max(config.THRESHOLD, (1 + config.MARGIN) / 2)
    hi = 1 - lo
    return f"""
    <div class="verdict {key}" role="status">
      <h2>{title}</h2>
      <p>{text}</p>
      <div class="scale" aria-hidden="true">
        <div class="scale-track">
          <span class="z-fresh"  style="width:{lo*100:.1f}%"></span>
          <span class="z-unsure" style="width:{(hi-lo)*100:.1f}%"></span>
          <span class="z-rotten" style="width:{(1-hi)*100:.1f}%"></span>
          <span class="marker" style="inset-inline-start:{rot*100:.1f}%"></span>
        </div>
        <div class="scale-labels"><span>طازجة</span><span>غير متأكد</span><span>تالفة</span></div>
      </div>
      <p class="scale-readout">احتمال التلف: <bdi dir="ltr">{rot:.0%}</bdi></p>
    </div>"""


# ---------------------------------------------------------------
# 2) الواجهة
# ---------------------------------------------------------------
st.markdown("""
<div class="head">
  <h1>فاحص طزاجة البندورة</h1>
  <p>صوّر حبة بندورة وحدة، والنموذج بيقلّك إذا طازجة أو تالفة، ووين شاف العلامات.</p>
</div>
""", unsafe_allow_html=True)

if is_placeholder:
    st.markdown('<div class="notice">هذا نموذج مؤقت غير مدرّب، فالنتيجة مش حقيقية. '
                'النموذج الحقيقي بينضاف بعد التدريب.</div>', unsafe_allow_html=True)

tab_check, tab_how, tab_limits = st.tabs(["الفحص", "كيف يعمل", "القيود"])

with tab_check:
    source = st.segmented_control("مصدر الصورة", ["الكاميرا", "صورة من الجهاز"],
                                  default="الكاميرا", label_visibility="collapsed")
    if source == "صورة من الجهاز":
        uploaded = st.file_uploader("اختر صورة لحبة بندورة",
                                    type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded = st.camera_input("صوّر حبة بندورة")

    with st.expander("نصائح لصورة أوضح"):
        st.markdown("- حبة **وحدة** تعبّي معظم الصورة\n"
                    "- إضاءة قوية بدون فلاش وبدون ظل على الحبة\n"
                    "- صوّر الجهة اللي فيها علامات، إذا في\n"
                    "- ثبّت إيدك: الصورة المهزوزة بتخلّي النموذج غير متأكد")

    if uploaded is not None:
        img, error = open_photo(uploaded)
        if error:
            st.error(error)
        else:
            with st.spinner("جاري الفحص..."):
                probs, cls, shown, overlay = analyze(img)
            st.markdown(verdict_html(probs, cls), unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.image(shown, caption="صورتك", width="stretch")
            if overlay is None:
                col2.markdown('<div class="notice">ما في منطقة بالصورة لفتت نظر النموذج أكثر من '
                              'غيرها، فما في خريطة نعرضها.</div>', unsafe_allow_html=True)
            else:
                col2.image(overlay, caption="وين ركّز النموذج (الأحمر = الأهم)", width="stretch")

with tab_how:
    st.markdown(f"""
#### كيف يعمل؟
1. **النموذج:** EfficientNet-B0 مدرّب مسبقاً على ImageNet، وأُعيد تدريبه ليفرّق بين
   حبة بندورة **طازجة** و**تالفة**.
2. **البيانات:** حوالي ألفين صورة أصلية لحبات بندورة من مجموعة Khatun وزملائها (2023)،
   مقسّمة بحيث الصور المتشابهة جداً ما تكون بالتدريب والاختبار مع بعض.
3. **"غير متأكد":** إذا الثقة أقل من {config.THRESHOLD:.0%}، أو الفرق بين الاحتمالين أقل من
   {config.MARGIN:.0%}، التطبيق ما بيعطي حكم. هذا هو الشريط المخطط بنص المقياس.
4. **وين ركّز؟** تقنية **Grad-CAM** بتلوّن المناطق اللي اعتمد عليها النموذج. إذا الأحمر على
   الخلفية بدل الحبة، لا تثق بالنتيجة.

نتائج الدقة بتنضاف هون بعد التدريب، على صور الاختبار وعلى صور صوّرناها بأنفسنا بالمحلات.
""")

with tab_limits:
    st.markdown("""
#### القيود (بصراحة)
- **بندورة بس.** أي خضرة أو فاكهة ثانية رح يحاول يصنّفها طازجة أو تالفة بالغلط.
- **حبة وحدة بالصورة.** إذا في كذا حبة، الحكم بيكون على الصورة كلها مش على حبة معيّنة.
- **ما بيشوف كل شي:** التلف من جوّا، الرضوض من الجهة الثانية، والطعم ما بيبيّنوا بالصورة.
- **صور التدريب** معظمها على قماش أبيض أو طاولة خشب، فالدقة بالمحل ممكن تكون أقل.
- **الثقة مش ضمان:** النموذج ممكن يكون واثق وغلطان، وهو مش بديل عن الفحص بالعين واليد.
""")
    st.caption("البيانات: Khatun, Razzak, Islam & Uddin (2023), Tomato Maturity Detection and "
               "Quality Grading Dataset, Mendeley Data, doi:10.17632/s42kpg8h37.1 — CC BY 4.0.")

st.divider()
st.markdown(f'<p class="foot">نموذج تجريبي لأغراض التعلّم. '
            f'<a href="{GITHUB_URL}">الكود على GitHub</a></p>', unsafe_allow_html=True)

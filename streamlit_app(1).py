"""
مشخّص أمراض أوراق البندورة — نموذج تجريبي
Tomato Leaf Disease Classifier (EfficientNet-B0 + Grad-CAM) — Streamlit
Author: Abdullah Asasfeh
"""
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from matplotlib import colormaps
from torchvision import models, transforms

st.set_page_config(page_title="مشخّص أمراض أوراق البندورة", page_icon="🍅",
                   layout="centered", initial_sidebar_state="collapsed")

GITHUB_URL = "https://github.com/abdullah-asasfeh/tomato-leaf-disease"

# ---------------------------------------------------------------
# 0) التصميم
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
html, body, .stApp { font-family: 'Tajawal', sans-serif; }
.stApp, .stMarkdown, p, h1, h2, h3, h4, label, li { direction: rtl; text-align: right; }

.hero { padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%); }
.hero h1 { color: #fff; margin: 0; font-size: 2rem; font-weight: 800; }
.hero p  { color: #e8f5e9; margin: .4rem 0 0; font-size: 1.05rem; }
.badges span { display: inline-block; background: rgba(255,255,255,.18); color: #fff;
               padding: .2rem .75rem; border-radius: 999px; font-size: .85rem;
               margin: .7rem 0 0 .4rem; }

.card { border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem 1.25rem; margin: .8rem 0; }
.card.ok   { border-right: 6px solid #2e7d32; background: rgba(46,125,50,.08); }
.card.warn { border-right: 6px solid #f9a825; background: rgba(249,168,37,.10); }
.card h3 { margin: 0 0 .35rem; font-size: 1.35rem; }
.card p  { margin: .2rem 0; }
.tag { display: inline-block; font-size: .8rem; padding: .1rem .6rem; border-radius: 999px;
       background: rgba(128,128,128,.18); margin-right: .4rem; }

.bar-row { margin: .6rem 0; }
.bar-label { display: flex; justify-content: space-between; font-size: .95rem; margin-bottom: .25rem; }
.bar { height: 10px; border-radius: 999px; background: rgba(128,128,128,.2);
       overflow: hidden; direction: rtl; }
.bar > div { height: 100%; border-radius: 999px; background: #43a047; }
.bar.first > div { background: #1b5e20; }

.foot { font-size: .85rem; opacity: .75; text-align: center !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 1) الأصناف بنفس ترتيب التدريب (class_to_idx)
# ---------------------------------------------------------------
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
]

# الاسم بالعربي، النوع، وصف الأعراض
INFO = {
    "Bacterial_spot": ("التبقع البكتيري", "مرض بكتيري",
        "بقع صغيرة داكنة على الأوراق، وينتشر مع الرطوبة ورذاذ الماء."),
    "Early_blight": ("اللفحة المبكرة", "مرض فطري",
        "بقع بنية فيها حلقات متداخلة، وغالباً يبدأ بالأوراق السفلية."),
    "Late_blight": ("اللفحة المتأخرة", "مرض فطري",
        "بقع كبيرة داكنة سريعة الانتشار، ويزداد في الجو البارد الرطب."),
    "Leaf_Mold": ("عفن الأوراق", "مرض فطري",
        "بقع صفراء على سطح الورقة ونمو زيتوني أو رمادي على السطح السفلي، ويرتبط بالرطوبة العالية."),
    "Septoria_leaf_spot": ("تبقّع الأوراق السبتوري", "مرض فطري",
        "بقع صغيرة دائرية بمركز فاتح وحافة داكنة، تبدأ غالباً بالأوراق السفلية."),
    "Spider_mites Two-spotted_spider_mite": ("العنكبوت الأحمر", "آفة حشرية",
        "تنقيط أصفر دقيق على الأوراق وأحياناً خيوط رفيعة، ويزداد في الجو الحار الجاف."),
    "Target_Spot": ("التبقّع الهدفي", "مرض فطري",
        "بقع بنية فيها حلقات تشبه الهدف."),
    "Tomato_Yellow_Leaf_Curl_Virus": ("فيروس تجعّد واصفرار الأوراق", "مرض فيروسي",
        "اصفرار وتجعّد والتفاف الأوراق للأعلى مع تقزّم النبتة، وينقله الذباب الأبيض."),
    "Tomato_mosaic_virus": ("فيروس الموزاييك", "مرض فيروسي",
        "تبرقش بين الأخضر الفاتح والغامق وتشوّه في الأوراق، وينتقل بالتلامس والأدوات."),
    "healthy": ("ورقة سليمة", "سليم",
        "لا تظهر على الورقة أعراض واضحة لأحد الأمراض التي يعرفها النموذج."),
}

IMG_SIZE = 224
THRESHOLD = 0.60   # أقل ثقة مقبولة
MARGIN = 0.20      # أقل فرق مقبول بين الاحتمال الأول والثاني

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------
# 2) تحميل الموديل مرة وحدة بس (cache)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load("best_field_model.pt", map_location="cpu"))
    m.eval()
    saved = {}

    def hook(module, inp, out):
        saved["act"] = out
        out.register_hook(lambda g: saved.__setitem__("grad", g))

    m.features[-1].register_forward_hook(hook)
    return m, saved


model, saved = load_model()


def predict(img):
    x = tf(img).unsqueeze(0)
    out = model(x)
    probs = F.softmax(out, dim=1)[0]
    cls = int(probs.argmax())

    # Grad-CAM
    model.zero_grad()
    out[0, cls].backward()
    weights = saved["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * saved["act"]).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear")[0, 0].detach().numpy()

    base = img.resize((IMG_SIZE, IMG_SIZE))
    heat = (colormaps["jet"](cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(base, Image.fromarray(heat), alpha=0.45)
    return probs.detach().numpy(), cls, overlay


def bars_html(p, k=3):
    rows = []
    for rank, i in enumerate(p.argsort()[::-1][:k]):
        name = INFO[CLASS_NAMES[i]][0]
        cls_attr = "bar first" if rank == 0 else "bar"
        rows.append(
            f'<div class="bar-row"><div class="bar-label"><span>{name}</span>'
            f'<span>{p[i]:.0%}</span></div>'
            f'<div class="{cls_attr}"><div style="width:{p[i]*100:.1f}%"></div></div></div>'
        )
    return "".join(rows)


# ---------------------------------------------------------------
# 3) الشريط الجانبي
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### عن المشروع")
    st.write("نموذج تعلّم عميق يصنّف 10 حالات لأوراق البندورة (9 أمراض + سليمة)، "
             "ويوضّح بخريطة حرارية أين نظر في الصورة.")
    c1, c2 = st.columns(2)
    c1.metric("دقة صور المختبر", "99.3%")
    c2.metric("دقة صور الحقل", "62%")
    st.markdown("**النموذج:** EfficientNet-B0  \n**البيانات:** PlantVillage + PlantDoc  \n"
                "**التفسير:** Grad-CAM")
    st.markdown(f"[الكود على GitHub]({GITHUB_URL})")
    st.caption("Developed by Abdullah Asasfeh")

# ---------------------------------------------------------------
# 4) الواجهة
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🍅 مشخّص أمراض أوراق البندورة</h1>
  <p>صوّر ورقة بندورة واحدة، والنموذج بيعطيك التشخيص المرجّح ويوضّح وين نظر بالصورة.</p>
  <div class="badges"><span>10 حالات</span><span>EfficientNet-B0</span><span>Grad-CAM</span></div>
</div>
""", unsafe_allow_html=True)

tab_diag, tab_how, tab_limits = st.tabs(["🔍 التشخيص", "🧠 كيف يعمل", "⚠️ القيود"])

with tab_diag:
    source = st.radio("مصدر الصورة", ["رفع صورة", "الكاميرا"], horizontal=True,
                      label_visibility="collapsed")
    if source == "رفع صورة":
        uploaded = st.file_uploader("ارفع صورة ورقة بندورة",
                                    type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded = st.camera_input("صوّر ورقة بندورة")

    with st.expander("💡 نصائح لصورة أوضح"):
        st.markdown("- ورقة **واحدة** تملأ معظم الصورة\n"
                    "- إضاءة النهار، بدون ظل قوي أو فلاش\n"
                    "- صوّر الوجه الذي تظهر عليه الأعراض\n"
                    "- تجنّب الصور المشوّشة أو البعيدة")

    if uploaded is not None:
        img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        with st.spinner("جاري التحليل..."):
            p, cls, overlay = predict(img)

        order = p.argsort()[::-1]
        conf = float(p[cls])
        gap = float(p[order[0]] - p[order[1]])
        name, kind, desc = INFO[CLASS_NAMES[cls]]

        if conf < THRESHOLD or gap < MARGIN:
            st.markdown(f"""
            <div class="card warn">
              <h3>🤔 النموذج غير متأكد</h3>
              <p>أقرب احتمال: <b>{name}</b> ({conf:.0%})، لكن الثقة غير كافية لإعطاء تشخيص.</p>
              <p>جرّب صورة أقرب لورقة واحدة بإضاءة النهار، أو راجع مهندساً زراعياً.</p>
            </div>""", unsafe_allow_html=True)
        else:
            icon = "✅" if CLASS_NAMES[cls] == "healthy" else "🩺"
            st.markdown(f"""
            <div class="card ok">
              <h3>{icon} {name} <span class="tag">{kind}</span></h3>
              <p>{desc}</p>
              <p style="opacity:.75">ثقة النموذج: {conf:.0%}</p>
            </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img.resize((IMG_SIZE, IMG_SIZE)), caption="الصورة", width=300)
        with col2:
            st.image(overlay, caption="أين نظر النموذج؟ (الأحمر = الأهم)", width=300)

        st.markdown("#### أعلى 3 احتمالات")
        st.markdown(bars_html(p), unsafe_allow_html=True)

with tab_how:
    st.markdown("""
#### كيف يعمل النموذج؟
1. **التدريب الأساسي:** نموذج EfficientNet-B0 مدرّب مسبقاً على ImageNet، ثم أُعيد تدريبه على
   حوالي 18 ألف صورة بندورة من مجموعة **PlantVillage** (صور مختبر بخلفية موحّدة) — دقة 99.3%.
2. **مشكلة الحقل:** عند اختباره على صور حقيقية من **PlantDoc** انخفضت الدقة إلى **30%** فقط،
   لأن النموذج تعوّد على خلفية المختبر (Domain Shift).
3. **التحسين:** أُضيفت صور الحقل إلى التدريب (Fine-tuning) فارتفعت دقة الحقل إلى **62%**
   مع بقاء دقة المختبر 99.3%.
4. **التفسير:** تقنية **Grad-CAM** تُظهر المناطق التي اعتمد عليها النموذج — عند الإجابات الصحيحة
   يركّز على بقع المرض، وعند الخطأ غالباً ينظر إلى الخلفية.

| النموذج | صور المختبر | صور الحقل |
|---|---|---|
| قبل التحسين | 99.3% | 30.4% |
| بعد التحسين | 99.3% | 62.0% |
""")

with tab_limits:
    st.markdown("""
#### القيود (بصراحة)
- **يعرف 10 حالات فقط** لأوراق البندورة؛ أي صورة أخرى (نبات آخر، ثمرة، صورة عشوائية)
  سيحاول تصنيفها ضمن هذه الحالات.
- **الثقة ليست ضماناً:** النماذج العميقة قد تعطي ثقة عالية وهي مخطئة، لذلك يعرض التطبيق
  "غير متأكد" عندما تكون الثقة أقل من 60% أو قريبة بين احتمالين.
- **دقة الحقل حوالي 62%**، ومجموعة اختبار الحقل صغيرة (79 صورة)، فالرقم تقريبي.
- بعض صور PlantDoc فيها أخطاء في التصنيف، وقد تكون صور الإنترنت الشائعة ضمن بيانات التدريب.
- **العنكبوت الأحمر** لم يُدرَّب على صور حقل له.
""")

st.divider()
st.markdown('<p class="foot">⚠️ نموذج تجريبي لأغراض التعلّم، وليس بديلاً عن تشخيص المهندس الزراعي.</p>',
            unsafe_allow_html=True)

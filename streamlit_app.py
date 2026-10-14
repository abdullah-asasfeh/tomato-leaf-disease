"""
مشخّص أمراض أوراق البندورة — نموذج تجريبي
Tomato Leaf Disease Classifier (EfficientNet-B0 + Grad-CAM) — Streamlit
"""
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps
from torchvision import models, transforms

st.set_page_config(page_title="مشخّص أمراض أوراق البندورة", page_icon="🍅", layout="wide")

# اتجاه الكتابة من اليمين لليسار
st.markdown("""
<style>
  .stApp, .stMarkdown, p, h1, h2, h3, label { direction: rtl; text-align: right; }
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

INFO = {
    "Bacterial_spot": ("التبقع البكتيري",
        "مرض بكتيري يظهر كبقع صغيرة داكنة على الأوراق، وينتشر مع الرطوبة ورذاذ الماء."),
    "Early_blight": ("اللفحة المبكرة",
        "مرض فطري يظهر كبقع بنية فيها حلقات متداخلة، وغالباً يبدأ بالأوراق السفلية."),
    "Late_blight": ("اللفحة المتأخرة",
        "مرض فطري سريع الانتشار، يظهر كبقع كبيرة داكنة، ويزداد في الجو البارد الرطب."),
    "Leaf_Mold": ("عفن الأوراق",
        "بقع صفراء على سطح الورقة ونمو زيتوني أو رمادي على السطح السفلي، ويرتبط بالرطوبة العالية."),
    "Septoria_leaf_spot": ("تبقّع الأوراق السبتوري",
        "بقع صغيرة دائرية بمركز فاتح وحافة داكنة، تبدأ غالباً بالأوراق السفلية."),
    "Spider_mites Two-spotted_spider_mite": ("العنكبوت الأحمر",
        "آفة تسبب تنقيطاً أصفر دقيقاً على الأوراق وأحياناً خيوطاً رفيعة، وتزداد في الجو الحار الجاف."),
    "Target_Spot": ("التبقّع الهدفي",
        "مرض فطري يظهر كبقع بنية فيها حلقات تشبه الهدف."),
    "Tomato_Yellow_Leaf_Curl_Virus": ("فيروس تجعّد واصفرار الأوراق",
        "اصفرار وتجعّد والتفاف الأوراق للأعلى مع تقزّم النبتة، وينقله الذباب الأبيض."),
    "Tomato_mosaic_virus": ("فيروس الموزاييك",
        "تبرقش بين الأخضر الفاتح والغامق وتشوّه في الأوراق، وينتقل بالتلامس والأدوات."),
    "healthy": ("ورقة سليمة",
        "لا تظهر على الورقة أعراض واضحة لأحد الأمراض التي يعرفها النموذج."),
}

IMG_SIZE = 224
THRESHOLD = 0.60  # تحت هذه الثقة نقول "غير متأكد"

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


# ---------------------------------------------------------------
# 3) الواجهة
# ---------------------------------------------------------------
st.title("🍅 مشخّص أمراض أوراق البندورة")
st.markdown(
    "ارفع صورة **ورقة بندورة واحدة** (من قريب وبإضاءة جيدة)، وسيعرض النموذج أعلى 3 احتمالات، "
    "وخريطة حرارية توضّح **أين نظر** النموذج في الصورة (الأحمر = المناطق الأهم)."
)

uploaded = st.file_uploader("صورة ورقة البندورة", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    with st.spinner("جاري التحليل..."):
        p, cls, overlay = predict(img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img.resize((IMG_SIZE, IMG_SIZE)), caption="الصورة", width=320)
    with col2:
        st.image(overlay, caption="أين نظر النموذج؟ (Grad-CAM)", width=320)

    conf = float(p[cls])
    name, desc = INFO[CLASS_NAMES[cls]]
    if conf < THRESHOLD:
        st.warning(f"🤔 النموذج غير متأكد ({conf:.0%}). صوّر ورقة بندورة واحدة من قريب "
                   "وبإضاءة النهار وحاول مرة ثانية، أو راجع مهندساً زراعياً.")
    else:
        st.success(f"التشخيص المرجّح: **{name}** ({conf:.0%})")
        st.write(desc)

    st.subheader("أعلى 3 احتمالات")
    for i in p.argsort()[::-1][:3]:
        st.write(f"{INFO[CLASS_NAMES[i]][0]} — {p[i]:.0%}")
        st.progress(float(p[i]))

st.divider()
st.caption(
    "⚠️ نموذج تجريبي لأغراض التعلّم، وليس بديلاً عن تشخيص المهندس الزراعي. "
    "دقته على صور الحقل الحقيقية حوالي 62%، ويتعرّف فقط على 10 حالات لأوراق البندورة."
)

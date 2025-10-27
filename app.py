import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.set_page_config(page_title="Deteksi Penyakit Mata", layout="centered")
st.title("🩺 Aplikasi Deteksi Penyakit Mata")

@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model/model_katarak.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

model = load_model()
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

uploaded_file = st.file_uploader("📤 Upload gambar mata", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.reshape(img_array, (1, 224, 224, 3))

    # Get model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Prediksi
    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])

    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_label = class_names[class_index]

    confidence = np.max(predictions[0]) * 100

    # Hasil
    with col2:
        st.subheader("📊 Hasil Prediksi")

        if predicted_label == "normal":
            st.success(f"**{predicted_label}** ({confidence:.2f}% yakin)")
            st.write("Mata Anda terlihat normal. Tetap jaga kesehatan mata!")
        else:
            st.error(f"**{predicted_label}** ({confidence:.2f}% yakin)")
            st.warning("Terdeteksi adanya indikasi penyakit. Harap segera konsultasi dengan dokter mata.")

    st.write("---")
    st.write("### Detail Probabilitas:")
    prob_data = {label: [f"{prob * 100:.2f}%"] for label, prob in zip(class_names, predictions[0])}
    st.dataframe(prob_data, use_container_width=True)

elif uploaded_file is None and model is not None:
    st.info("Silakan upload gambar untuk memulai deteksi.")
# CSS NAVBAR

st.markdown("""
    <style>
    div[data-testid="column"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    button[kind="secondary"] {
        background-color: #1E1E1E !important;
        color: #ccc !important;
        border-radius: 20px !important;
        border: none !important;
        margin: 0 5px !important;
        font-weight: 500 !important;
        transition: 0.3s;
    }
    button[kind="secondary"]:hover {
        background-color: #111 !important;
        color: white !important;
    }
    .active-btn {
        background-color: #00BFFF !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# STATE NAVBAR
# -----------------------------
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Cataract"

tabs = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# -----------------------------
# NAVBAR INTERAKTIF (PAKE ST COLUMNS)
# -----------------------------
cols = st.columns(len(tabs))
for i, tab in enumerate(tabs):
    btn_style = "active-btn" if st.session_state.selected_tab == tab else ""
    if cols[i].button(tab, key=tab, use_container_width=True):
        st.session_state.selected_tab = tab
    st.markdown(f"""
        <style>
        div[data-testid="stButton"] button[key="{tab}"] {{
            background-color: {'#00BFFF' if btn_style else '#1E1E1E'} !important;
            color: {'white' if btn_style else '#ccc'} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

st.write("---")
# -----------------------------
# KONTEN PENYAKIT BERDASARKAN TAB
# -----------------------------
tab = st.session_state.selected_tab

if tab == "Cataract":
    st.title("Cataract (Katarak)")
    st.write("""
    **Katarak** terjadi ketika lensa mata menjadi keruh sehingga cahaya tidak bisa masuk dengan baik ke retina.

    **Gejala umum:**
    - Penglihatan buram seperti berkabut  
    - Warna terlihat pudar  
    - Penurunan penglihatan di malam hari 
    - Sensitif terhadap cahaya  
    
    **Penyebab**
    - Penuaan: Proses degenerasi protein pada lensa mata
    - Diabetes: Penyakit sistemik yang meningkatkan risiko
    - Paparan Sinar UV: Paparan sinar ultraviolet berlebihan
    - Cedera atau peradangan mata: Riwayat cedera atau peradangan pada mata
    - Penggunaan obat-obatan tertentu: Penggunaan kortikosteroid jangka panjang
    - Faktor keturunan: Riwayat keluarga
    - Merokok: Meningkatkan risiko katarak secara signifikan
    - Konsumsi alkohol berlebihan: Juga merupakan salah satu faktor risiko 
             

    **Pencegahan:**
    - Mengonsumsi makanan kaya antioksidan seperti lutein dan zeaxanthin (dari sayuran yang diolah hijau tua).   
    - Menjaga asupan vitamin C dan E yang cukup dari buah-buahan dan kacang-kacangan.   
    - Menghentikan kebiasaan merokok. 
    - Menggunakan kacamata pelindung untuk menghindari paparan sinar UV.   
    """)

elif tab == "Diabetic Retinopathy":
    st.title("Diabetic Retinopathy")
    st.write("""
    Retinopati diabetik adalah kerusakan pada pembuluh darah retina mata akibat kadar gula darah tinggi yang tidak terkontrol pada penderita diabetes.

    **Gejala umum:**
    - Pada tahap awal, seringkali tidak ada gejala.  
    - Penglihatan menjadi buram.  
    - Muncul bintik-bintik atau “bayangan” gelap di mata.
    - Distorsi penglihatan (garis lurus tampak bergelombang).
    - Penurunan penglihatan yang signifikan tanpa rasa sakit. 

    **Penyebab**
    - Kadar gula darah tinggi yang tidak terkontrol dalam jangka waktu lama.
    - Kerusakan pada pembuluh darah di retina mata akibat komplikasi diabetes melitus.
    - Faktor risiko lain seperti tekanan darah tinggi dan kolesterol tinggi dapat menghemat kondisi. 
    
    **Pencegahan:**
    - Kontrol kadar gula darah  
    - Rutin periksa mata setiap tahun  
    - Hindari tekanan darah tinggi  
    """)

elif tab == "Glaucoma":
    st.title("Glaucoma (Glaukoma)")
    st.write("""
    **Glaukoma** adalah penyakit mata yang menyebabkan kerusakan pada saraf optik, seringkali akibat tekanan tinggi di dalam bola mata.

    **Gejala umum:**
    - Penglihatan sisi berkurang  
    - Sakit kepala dan nyeri mata  
    - Melihat lingkaran cahaya di sekitar lampu  
    
    **Penyebab**
    - Glaukoma terjadi ketika tekanan cairan di dalam bola mata (aqueous humor) meningkat dan merusak saraf optik, yang berfungsi mengirimkan informasi visual dari mata ke otak. 
    - Peningkatan tekanan ini bisa disebabkan oleh produksi cairan yang berlebihan atau penyumbatan pada saluran drainase (trabecular meshwork). 
    - Faktor risiko lain seperti tekanan darah tinggi dan kolesterol tinggi dapat menghemat kondisi. 

    **Pencegahan:**
    - Tidak dapat dibudidayakan, tetapi dapat dikelola: Glaukoma tidak bisa disembuhkan, namun kerusakan yang sudah terjadi tidak dapat diperbaiki. Tujuannya adalah memperlambat perkembangan penyakit dan mencegah kebutaan lebih lanjut. 
    - Obat tetes mata: Untuk mengurangi produksi cairan atau memperlancar alirannya.   
    - Terapi laser: Membantu membuka saluran drainase. 
    - Operasi: Membuat saluran drainase baru jika pengobatan lain tidak efektif.  
    """)

elif tab == "Normal":
    st.title("Mata Normal")
    st.write("""
    Mata dalam kondisi **normal** berarti tidak ditemukan tanda-tanda kelainan atau penyakit serius.

    **Ciri-ciri:**
    - Penglihatan jelas  
    - Tidak nyeri atau merah  
    - Tidak sensitif terhadap cahaya  

    **Tips menjaga kesehatan mata:**
    - Aturan 20-20-20 untuk istirahat mata  
    - Konsumsi makanan kaya vitamin A  
    - Pemeriksaan rutin tiap 6–12 bulan  
    """)

st.write("---")
st.caption("© 2025 Edukasi Penyakit Mata - Capstone Project")

import cv2
from ultralytics import YOLO
from pytesseract import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz  # PyMuPDF
import os
from translate import Translator
from symspellpy import SymSpell
import string

# Symspell dil sözlüğünü yükle
dictionary_path = "frequency_dictionary_en_82_765.txt"
symspell = SymSpell()
symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Önceden eğitilmiş bir YOLOv8n modelini yükleme
model = YOLO('metin-seg.pt')

font = ImageFont.truetype("arial.ttf", 9)

# Tesseract kurulum yolu
path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.tesseract_cmd = path_to_tesseract

# PDF dosyasını yükle
pdf_path = 'HedefManga.pdf'
pdf_document = fitz.open(pdf_path)

# Çıktı PDF dosyasının adını ve geçici klasör
output_pdf_path = 'TranslatedManga.pdf'
temp_folder = 'temp_images'

# Geçici klasör yoksa oluştur
os.makedirs(temp_folder, exist_ok=True)

# translate kütüphanesi ile Translator nesnesini oluşturma
translator = Translator(to_lang="tr")

def clean_text(text):
    # Gereksiz boşlukları ve satır sonlarını temizleme
    text = text.strip()
    text = ' '.join(text.split())
    # Fazla noktalama işaretlerini kaldırma
    text = ' '.join(word.strip(string.punctuation) for word in text.split())
    # Metni küçük harfe dönüştürme
    text = text.lower()
    return text

def correct_spelling(text):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        corrected_text = suggestions[0].term
        return corrected_text
    else:
        return text

def process_page(page_num):
    # Sayfayı görüntüye dönüştür
    page = pdf_document.load_page(page_num)
    pixmap = page.get_pixmap()
    image_pil = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
    image = np.array(image_pil)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    draw = ImageDraw.Draw(image_pil)

    # YOLO modelini kullanarak metin alanlarını tespit etme
    results = model.predict(image_rgb, show=False, conf=0.4, save=False)
    xyxy = results[0].boxes.xyxy.tolist()
    masks = results[0].masks

    # Eğer maske yoksa, işlemi sonlandır
    if masks is None:
        return

    # Tüm maskeler ve poligon koordinatlarını al
    all_polygons = [mask.xy[0].tolist() for mask in masks]

    # Önce tüm metin alanlarını beyaza boyama
    for polygon in all_polygons:
        polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon, fill="white")

    # Tüm metin alanlarını beyaza boyadıktan sonra, metinleri çıkarma ve yapıştırma
    for box in xyxy:
        x1, y1, x2, y2 = box
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        roi = image[iy1:iy2, ix1:ix2]

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(roi, config=custom_config)
        clean_text_input = clean_text(text)
        print(f"OCR Text: {clean_text_input}")
        corrected_text = correct_spelling(clean_text_input)
        print(f"Corrected Text: {corrected_text}")
        translation = translator.translate(corrected_text)
        print(f"Translated Text: {translation}")

        # Metni iki kelimede bir alt satıra geçecek şekilde düzenleme
        words = translation.split()
        formatted_text = "\n".join([" ".join(words[i:i + 2]) for i in range(0, len(words), 2)])

        draw.text((x1, y1), formatted_text, font=font, fill=(0, 0, 0, 255))

    # Güncellenmiş sayfayı geçici klasöre kaydetme
    temp_image_path = os.path.join(temp_folder, f'page_{page_num + 1}.png')
    image_pil.save(temp_image_path, 'PNG')

    # Belleği serbest bırak
    del image_pil
    del image
    del image_rgb

# Her sayfayı işleme
for page_num in range(len(pdf_document)):
    process_page(page_num)

# Geçici klasördeki resimleri yeni bir PDF dosyasına birleştirme
image_files = [os.path.join(temp_folder, f) for f in sorted(os.listdir(temp_folder)) if f.endswith('.png')]
image_list = [Image.open(img_file) for img_file in image_files]
image_list[0].save(output_pdf_path, save_all=True, append_images=image_list[1:], resolution=100.0)

# Geçici klasörü temizleme
for img_file in image_files:
    os.remove(img_file)

os.rmdir(temp_folder)

print("PDF dosyası başarıyla oluşturuldu.")

import os  # Dosya ve dizin işlemleri için
import re  # Düzenli ifadeler için
import urllib.parse  # URL ayrıştırma için
import warnings  # Uyarıları yönetmek için
from heapq import nlargest  # En büyük n öğeyi bulmak için
from PyQt5.QtWidgets import (
    QApplication, QGroupBox, QMainWindow, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QSlider, QHBoxLayout, QVBoxLayout,
    QWidget, QCheckBox # PyQt5 widget'ları
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal # PyQt5 çekirdek modülleri
import requests  # HTTP istekleri için
import pandas as pd  # Veri işleme için
import numpy as np  # Sayısal işlemler için
import networkx as nx  # Grafik algoritmaları için
import pdfplumber  # PDF dosyalarını işlemek için
from langdetect import detect  # Dil tespiti için
from nltk.tokenize import word_tokenize, sent_tokenize  # Metni tokenleştirme için
from nltk.corpus import stopwords  # Stop kelimeleri için
from nltk.cluster.util import cosine_distance  # Kosinüs benzerliği için
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vektörleştirmesi için
from sklearn.metrics.pairwise import cosine_similarity  # Kosinüs benzerliği için
import torch  # PyTorch kütüphanesi
from transformers import AutoTokenizer, AutoModel  # Transformer modelleri için
from summarizer import Summarizer  # Özetleme için
from selenium import webdriver  # Web tarayıcısı otomasyonu için
from selenium.webdriver.common.by import By  # Web öğelerini bulmak için
from selenium.webdriver.support.ui import WebDriverWait  # Web öğelerini beklemek için
from selenium.webdriver.support import expected_conditions as EC  # Beklenen koşullar için

warnings.filterwarnings("ignore", category=FutureWarning)  # FutureWarning uyarılarını yoksay

# Global değişkenler
sonuclar_df = pd.DataFrame(
    columns=['Referans Cümle', 'Benzerlik Oranı', 'Cümle', 'Kaynak']
)  # Sonuçları saklamak için DataFrame

# Türkçe BERT modelini yükle (BERTSUM ve TEXTRANK için)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased") # Tokenizer
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased") # Model

# BERTSUM modelini yükle
bertsum_model = Summarizer()  # BERTSUM özetleyici


class WorkerThread(QThread): # Arka planda çalışacak iş parçacığı
    progress_updated = pyqtSignal(int) # İlerleme güncelleme sinyali
    results_ready = pyqtSignal(str) # Sonuçlar hazır sinyali
    error_occurred = pyqtSignal(str) # Hata oluştu sinyali

    def __init__(self, keyword, link_count, benzerlik_esigi, secilen_algoritmalar, referans_metin): # Yapıcı metot
        super().__init__() # Üst sınıfın yapıcısını çağır
        self.keyword = keyword # Anahtar kelime
        self.link_count = link_count # İndirilecek makale sayısı
        self.benzerlik_esigi = benzerlik_esigi # Benzerlik eşiği
        self.secilen_algoritmalar = secilen_algoritmalar # Seçilen özetleme algoritmaları
        self.referans_metin = referans_metin # Referans metin


    def run(self): # İş parçacığının çalıştırılacağı metot
        try:  # Hata yönetimi için try-except bloğu
            pdf_links = self.scholar_pdf_links_al(self.keyword, self.link_count) # PDF linklerini al
            total_links = len(pdf_links)  # Toplam link sayısı

            for i, link in enumerate(pdf_links):  # Her link için döngü
                makale_linki = link # Makale linki
                pdf_text = self.pdf_indir_ve_isle(link, self.secilen_algoritmalar) # PDF'yi indir ve işle
                if pdf_text: # PDF metni varsa
                    makale_cumleler = sent_tokenize(pdf_text) # Cümlelere ayır
                    self.kaynak_bul(self.benzerlik_esigi, makale_cumleler, makale_linki)  # Benzer kaynakları bul

                # İlerlemeyi güncelle
                progress = int((i + 1) / total_links * 100)  # İlerleme yüzdesi
                self.progress_updated.emit(progress)   # İlerleme sinyalini gönder

            # Sonuçları kaydet
            dosya_adi = "sonuclar.xlsx"  # Dosya adı
            klasor_yolu = os.path.join(os.path.expanduser("~"), "Desktop")   # Masaüstü yolu
            tam_dosya_yolu = os.path.join(klasor_yolu, dosya_adi)   # Tam dosya yolu

            if not os.path.exists(klasor_yolu):  # Klasör yoksa oluştur
                os.makedirs(klasor_yolu)   # Klasör oluştur

            sonuclar_df.to_excel(tam_dosya_yolu, index=False)  # Sonuçları Excel dosyasına kaydet

            self.results_ready.emit("İşlem tamamlandı!")   # İşlem tamamlandı sinyalini gönder

        except Exception as e:  # Hata yakalama
            self.error_occurred.emit(f"Hata oluştu: {str(e)}")  # Hata sinyalini gönder


    def metni_temizle(self, text): # Metni temizleme fonksiyonu
        text = self.remove_brackets(text) # Parantezleri kaldır
        text = text.lower()  # Küçük harfe çevir
        text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
        tokens = word_tokenize(text)   # Kelimelere ayır
        stop_words = set(stopwords.words('turkish'))  # Türkçe stop kelimelerini yükle
        tokens = [
            word for word in tokens
            if word not in stop_words and len(word) > 2  # Stop kelimeleri ve kısa kelimeleri kaldır
        ]
        return " ".join(tokens) # Kelimeleri birleştirerek döndür


    def remove_brackets(self, text): # Parantezleri kaldırma fonksiyonu
        text = re.sub(r'\([^()]*\)', '', text)   # (...) parantezlerini kaldır
        text = re.sub(r'\[[^\[\]]*\]', '', text)  # [...] parantezlerini kaldır
        text = re.sub(r'\{[^{}]*\}', '', text)   # {...} parantezlerini kaldır
        return text # Temizlenmiş metni döndür


    def pdf_to_text(self, file_path):  # PDF'yi metne dönüştürme fonksiyonu
        try:  # Hata yönetimi için try-except bloğu
            with pdfplumber.open(file_path) as pdf:  # PDF dosyasını aç
                text = ""  # Metni saklamak için boş bir string
                for page in pdf.pages[1:-1]: # giriş ve kaynakça sayfalarını hariç tutarak sayfalarda döngü
                    page_text = page.extract_text() # Sayfa metnini çıkar
                    if "Giriş" in page_text: # giriş bölümü varsa
                        intro_index = page_text.find("Giriş") # giriş bölümünün başlangıç indeksini bul
                        text += page_text[intro_index:] # giriş bölümünden sonrasını metne ekle
                    elif "Kaynaklar" in page_text:  # kaynaklar bölümü varsa
                        break  # Döngüyü sonlandır
                    else:  # diğer sayfalar için
                        text += page_text + " " # sayfa metnini metne ekle

            return text.strip()  # Metnin başındaki ve sonundaki boşlukları temizle ve döndür

        except Exception as e:  # Hata yakalama
            self.error_occurred.emit(f"Dosya okunurken hata oluştu: {e}")   # Hata sinyalini gönder
            return ""  # Boş string döndür


    def get_sentence_embeddings(self, sentences): # Cümle gömmelerini alma fonksiyonu
        inputs = tokenizer(
            sentences, # Cümleler
            return_tensors="pt",  # PyTorch tensörleri olarak döndür
            padding=True, # Dolgulama yap
            truncation=True,  # Kırpma yap
            max_length=2048  # Maksimum uzunluk
        )
        with torch.no_grad(): # Gradyan hesaplama
            outputs = model(**inputs)  # Modeli çalıştır
        return outputs.last_hidden_state[:, 0, :].numpy()  # Son gizli katmanın çıktısını döndür


    def ozetleme_yap(self, metin, algoritma, max_ratio=1): # Özetleme fonksiyonu
        sentences = sent_tokenize(metin) # Cümlelere ayır
        max_sentences = max(1, int(len(sentences) * max_ratio)) # Maksimum cümle sayısı

        if algoritma == "BERTSUM": # BERTSUM algoritması seçildiyse
            summary = self.bertsum_summarize(metin, max_sentences)  # BERTSUM ile özetle
        elif algoritma == "TEXTRANK": # TEXTRANK algoritması seçildiyse
            summary = self.textrank_summarize(sentences, max_sentences)  # TEXTRANK ile özetle
        else:  # LEXRANK algoritması seçildiyse
            summary = self.lexrank_summarize(sentences, max_sentences)  # LEXRANK ile özetle

        return summary # Özeti döndür


    def bertsum_summarize(self, text, num_sentences): # BERTSUM özetleme fonksiyonu
        # BERTSUM ile özetleme işlemi
        result = bertsum_model(text, num_sentences=num_sentences) # BERTSUM modelini çalıştır
        summary = "".join(result).replace(" ", " ")  # Boşlukları düzelt
        return summary.replace("\n", " ")  # Tek satıra indirge


    def textrank_summarize(self, sentences, num_sentences): # TEXTRANK özetleme fonksiyonu
        # Cümleleri vektörleştir
        sentence_vectors = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt") # Cümle vektörlerini al
        with torch.no_grad(): # Gradyan hesaplama
            sentence_embeddings = model(**sentence_vectors).last_hidden_state[:, 0, :].numpy() # Cümle gömmelerini al

        # Benzerlik matrisini oluştur
        sim_mat = np.zeros([len(sentences), len(sentences)])  # Benzerlik matrisi
        for i in range(len(sentences)):  # Cümlelerde döngü
            for j in range(len(sentences)):  # Cümlelerde döngü
                if i != j:  # Farklı cümleler için
                    sim_mat[i][j] = 1 - cosine_distance(sentence_embeddings[i], sentence_embeddings[j]) # Kosinüs benzerliğini hesapla

        # PageRank algoritmasını uygula
        nx_graph = nx.from_numpy_array(sim_mat)  # Grafik oluştur
        scores = nx.pagerank(nx_graph)  # PageRank skorlarını hesapla

        # En yüksek skorlu cümleleri seç
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)  # Skorlara göre sırala

        # Özetleme için en iyi cümleleri seç
        summary_sentences = []  # Özet cümleleri
        for i in range(min(num_sentences, len(ranked_sentences))): # En yüksek skorlu cümlelerde döngü
            summary_sentences.append(ranked_sentences[i][1]) # Cümleyi ekle

        # Cümleleri orijinal sırasına göre sırala
        summary_sentences.sort(key=lambda x: sentences.index(x)) # Orijinal sıraya göre sırala

        summary = " ".join(summary_sentences)  # Cümleleri birleştir
        return summary.replace("\n", " ")  # Alt paragrafları kaldır


    def lexrank_summarize(self, sentences, num_sentences):  # LEXRANK özetleme fonksiyonu
        tfidf_vectorizer = TfidfVectorizer() # TF-IDF vektörleştirici
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences) # TF-IDF matrisini oluştur

        # Cümleler arası benzerlik matrisini oluştur
        sim_mat = cosine_similarity(tfidf_matrix, tfidf_matrix) # Kosinüs benzerlik matrisini oluştur

        # LexRank skorlarını hesapla
        nx_graph = nx.from_numpy_array(sim_mat) # Grafik oluştur
        scores = nx.pagerank(nx_graph)   # PageRank skorlarını hesapla

        # En yüksek skorlu cümleleri seç
        ranked_sentences = sorted(
            (((scores[i], s) for i, s in enumerate(sentences))), # Skorlara göre sırala
            reverse=True  # Ters sıralama
        )
        summary = " ".join(
            [ranked_sentences[i][1]  # Cümleyi ekle
             for i in range(min(num_sentences, len(ranked_sentences)))] # En yüksek skorlu cümlelerde döngü
        )
        return summary.replace("\n", " ")  # Alt paragrafları kaldır


    def sentence_similarity(self, sent1, sent2):  # Cümle benzerliği hesaplama fonksiyonu
        embedding1 = self.get_sentence_embeddings([sent1])[0] # İlk cümlenin gömmelerini al
        embedding2 = self.get_sentence_embeddings([sent2])[0]  # İkinci cümlenin gömmelerini al
        return 1 - cosine_distance(embedding1, embedding2)  # Kosinüs benzerliğini hesapla ve döndür


    def tfidf_benzerlik(self, referans_cumleler, makale_cumleler): # TF-IDF benzerliği hesaplama fonksiyonu
        tfidf_vectorizer = TfidfVectorizer()  # TF-IDF vektörleştirici
        all_sentences = referans_cumleler + makale_cumleler  # Tüm cümleleri birleştir
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_sentences) # TF-IDF matrisini oluştur
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Kosinüs benzerlik matrisini oluştur
        return cosine_sim[len(referans_cumleler):, :len(referans_cumleler)]  # Benzerlik matrisini döndür


    def kaynak_bul(self, benzerlik_esigi, makale_cumleler, makale_linki):  # Benzer kaynakları bulma fonksiyonu
        global sonuclar_df # Global sonuç DataFrame'i
        referans_cumleler = sent_tokenize(self.referans_metin)   # Referans metni cümlelere ayır

        cosine_sim = self.tfidf_benzerlik(referans_cumleler, makale_cumleler) # TF-IDF benzerliğini hesapla

        for i, referans_cumle in enumerate(referans_cumleler):  # Referans cümlelerde döngü
            max_tfidf_benzerlik = np.max(cosine_sim[:, i])  # Maksimum TF-IDF benzerliğini bul

            if max_tfidf_benzerlik >= benzerlik_esigi:  # Benzerlik eşiğini geçiyorsa
                benzerlik_index = np.argmax(cosine_sim[:, i]) # En benzer cümlenin indeksini bul
                benzerlik_orani = max_tfidf_benzerlik * 100  # Benzerlik oranını hesapla
                self.results_ready.emit( # Sonuç sinyalini gönder
                    f"Referans cümle {i + 1}: {referans_cumleler[i]}: " # Referans cümle
                    f"%{benzerlik_orani:.2f} benzerlik\n" # Benzerlik oranı
                )
                self.results_ready.emit( # Sonuç sinyalini gönder
                    f"Kaynak: {makale_linki} - Cümle {benzerlik_index + 1}: " # Kaynak ve cümle numarası
                    f"{makale_cumleler[benzerlik_index]}\n\n" # Benzer cümle
                )

                sonuclar_df.loc[len(sonuclar_df)] = [ # Sonuçları DataFrame'e ekle
                    referans_cumle,  # Referans cümle
                    benzerlik_orani,  # Benzerlik oranı
                    makale_cumleler[benzerlik_index],  # Benzer cümle
                    makale_linki  # Kaynak linki
                ]


    def generate_scholar_link(self, keyword):  # Google Scholar linki oluşturma fonksiyonu
        return (
            f"https://scholar.google.com/scholar?q={keyword}" # Anahtar kelime ile arama yap
            f"&hl=en&as_sdt=0,5"  # Arama parametreleri
        )


    def scholar_pdf_links_al(self, keyword, link_count): # Google Scholar'dan PDF linklerini alma fonksiyonu
        driver = webdriver.Chrome()  # Chrome WebDriver'ı başlat
        collected_links = 0  # Toplanan link sayacı
        link_list = []   # Linkleri saklamak için liste

        for i in range(10):  # Maksimum 10 sayfa ara
            url = self.generate_scholar_link(keyword)  # Google Scholar linkini oluştur
            driver.get(url)  # Sayfayı aç

            try:  # Hata yönetimi için try-except bloğu
                WebDriverWait(driver, 30).until( # 30 saniye bekle
                    EC.presence_of_element_located((  # PDF linklerini bul
                        By.XPATH,  # XPATH kullanarak
                        "//span[@class='gs_ctg2' and contains(text(), '[PDF]')]" # PDF linklerini içeren span etiketi
                        "/ancestor::a" # Link etiketi
                    ))
                )

                links = driver.find_elements(  # PDF linklerini bul
                    "xpath",   # XPATH kullanarak
                    "//span[@class='gs_ctg2' and contains(text(), '[PDF]')]" # PDF linklerini içeren span etiketi
                    "/ancestor::a" # Link etiketi
                )

                for link in links: # Linklerde döngü
                    href = link.get_attribute("href") # Linki al
                    if href: # Link varsa
                        link_list.append(href)  # Listeye ekle
                        collected_links += 1 # Sayacı artır
                        if collected_links >= link_count: # İstenilen sayıda link toplandıysa
                            break  # Döngüyü sonlandır
                if collected_links >= link_count:  # İstenilen sayıda link toplandıysa
                    break # Döngüyü sonlandır
            except Exception as e:  # Hata yakalama
                self.error_occurred.emit(f"Hata: {e}\n\n")  # Hata sinyalini gönder
                driver.quit() # WebDriver'ı kapat
                break  # Döngüyü sonlandır

        driver.quit()  # WebDriver'ı kapat
        return link_list  # Link listesini döndür


    def pdf_indir_ve_isle(self, url, ozetleme_algoritmalari): # PDF'yi indirme ve işleme fonksiyonu
        try: # Hata yönetimi için try-except bloğu
            response = requests.get(url) # PDF'yi indir
            response.raise_for_status()   # Hata durumunda hata fırlat

            klasor_yolu = os.path.join(os.path.expanduser("~"), "Desktop") # Masaüstü yolu
            makale_adi = os.path.basename(  # Makale adını al
                urllib.parse.urlparse(url).path # URL'den dosya adını al
            )

            if not makale_adi.endswith((".pdf")): # Dosya adı .pdf ile bitmiyorsa
                makale_adi += ".pdf"  # .pdf ekle

            pdf_dosya_yolu = os.path.join(klasor_yolu, makale_adi) # PDF dosya yolu
            with open(pdf_dosya_yolu, "wb") as f:  # PDF dosyasını kaydet
                f.write(response.content)  # Dosya içeriğini yaz

            pdf_text = self.pdf_to_text(pdf_dosya_yolu) # PDF'yi metne dönüştür

            # Dil tespiti
            try: # Hata yönetimi için try-except bloğu
                dil = detect(pdf_text)  # Metnin dilini tespit et
            except: # Hata durumunda
                dil = "unknown" # Bilinmiyor olarak işaretle

            for algoritma in ozetleme_algoritmalari: # Özetleme algoritmalarında döngü
                ozet = self.ozetleme_yap(pdf_text, algoritma) # Özetle
                txt_dosya_adi = (  # Metin dosya adını oluştur
                    os.path.splitext(makale_adi)[0] + # Dosya adı
                    f"_{algoritma}_{dil}.txt" # Algoritma ve dil bilgisi
                )
                txt_dosya_yolu = os.path.join(klasor_yolu, txt_dosya_adi)  # Metin dosya yolu
                with open(txt_dosya_yolu, "w", encoding="utf-8") as f:  # Metin dosyasını kaydet
                    f.write(
                        f"makale adı: {makale_adi}\n" # Makale adı
                        f"dil: {dil}\n" # Dil
                        f"özet ({algoritma}): {ozet}" # Özet
                    )

            return pdf_text # PDF metnini döndür
        except Exception as e:  # Hata yakalama
            self.error_occurred.emit( # Hata sinyalini gönder
                f"PDF indirme, işleme veya özetleme hatası: {e}\n\n" # Hata mesajı
            )
            return None  # None döndür


class KaynakBulmaArac(QMainWindow): # Ana pencere sınıfı
    def __init__(self): # Yapıcı metot
        super().__init__()  # Üst sınıfın yapıcısını çağır

        self.setWindowTitle("Kaynak Bulma Aracı")  # Pencere başlığı
        self.setGeometry(100, 100, 800, 600)  # Pencere boyutu

        self.central_widget = QWidget()  # Merkezi widget
        self.setCentralWidget(self.central_widget)   # Merkezi widget'ı ayarla

        self.layout = QVBoxLayout(self.central_widget)  # Dikey layout

        # Referans Metin
        self.label_referans = QLabel("Arama Yapılacak Metin:") # Etiket
        self.layout.addWidget(self.label_referans)  # Etiketi layout'a ekle

        self.text_referans = QTextEdit()  # Metin düzenleyici
        self.layout.addWidget(self.text_referans) # Metin düzenleyiciyi layout'a ekle

        # Anahtar Kelime
        self.label_keyword = QLabel("Anahtar Kelime:")  # Etiket
        self.layout.addWidget(self.label_keyword) # Etiketi layout'a ekle

        self.entry_keyword = QLineEdit()  # Metin girişi
        self.layout.addWidget(self.entry_keyword)   # Metin girişini layout'a ekle

        # İstenilen Makale Sayısı
        self.label_makale_sayisi = QLabel("İndirilecek Makale Sayısı:")  # Etiket
        self.layout.addWidget(self.label_makale_sayisi) # Etiketi layout'a ekle

        self.spinbox_makale_sayisi = QSpinBox()  # Sayı girişi
        self.spinbox_makale_sayisi.setMinimum(1) # Minimum değer
        self.spinbox_makale_sayisi.setMaximum(100)  # Maksimum değer
        self.layout.addWidget(self.spinbox_makale_sayisi)  # Sayı girişini layout'a ekle

        # Benzerlik Eşiği
        esik_layout = QHBoxLayout() # Yatay layout
        self.layout.addLayout(esik_layout) # Yatay layout'u dikey layout'a ekle

        self.label_esik = QLabel("Benzerlik Eşiği (%):")  # Etiket
        esik_layout.addWidget(self.label_esik) # Etiketi yatay layout'a ekle

        self.esik_degeri_label = QLabel("75") # Etiket
        esik_layout.addWidget(self.esik_degeri_label)  # Etiketi yatay layout'a ekle

        self.slider_esik = QSlider(Qt.Horizontal) # Yatay slider
        self.slider_esik.setRange(50, 100) # Slider aralığı
        self.slider_esik.setValue(75)  # Başlangıç değeri
        self.layout.addWidget(self.slider_esik)  # Slider'ı dikey layout'a ekle

        # Slider'ın değerini etikette göster
        self.slider_esik.valueChanged.connect(self.esik_degerini_guncelle) # Slider değeri değiştiğinde fonksiyonu çağır

        self.ozetleme_groupbox = QGroupBox("Özetleme Algoritması Seç") # Grup kutusu
        self.layout.addWidget(self.ozetleme_groupbox)  # Grup kutusunu dikey layout'a ekle

        self.checkbox_layout = QHBoxLayout()  # Yatay layout
        self.ozetleme_groupbox.setLayout(self.checkbox_layout) # Yatay layout'u grup kutusuna ekle

        self.checkbox_bertsum = QCheckBox("BERTSUM")  # Onay kutusu
        self.checkbox_bertsum.setChecked(True) # Başlangıçta seçili
        self.checkbox_layout.addWidget(self.checkbox_bertsum) # Onay kutusunu yatay layout'a ekle

        self.checkbox_textrank = QCheckBox("TEXTRANK") # Onay kutusu
        self.checkbox_layout.addWidget(self.checkbox_textrank) # Onay kutusunu yatay layout'a ekle

        self.checkbox_lexrank = QCheckBox("LEXRANK")  # Onay kutusu
        self.checkbox_layout.addWidget(self.checkbox_lexrank) # Onay kutusunu yatay layout'a ekle

        # Çalıştır Butonu
        self.button_calistir = QPushButton("Kaynakları Bul")  # Buton
        self.button_calistir.clicked.connect(self.button_calistir_command) # Butona tıklandığında fonksiyonu çağır
        self.layout.addWidget(self.button_calistir)  # Butonu dikey layout'a ekle

        # Sonuçlar
        self.label_sonuc = QLabel("Sonuç:") # Etiket
        self.layout.addWidget(self.label_sonuc) # Etiketi dikey layout'a ekle

        self.text_sonuc = QTextEdit()  # Metin düzenleyici
        self.layout.addWidget(self.text_sonuc)  # Metin düzenleyiciyi dikey layout'a ekle

        # İlerleme Çubuğu
        self.progress_bar = QProgressBar()  # İlerleme çubuğu
        self.progress_bar.setValue(0)   # Başlangıç değeri
        self.layout.addWidget(self.progress_bar)  # İlerleme çubuğunu dikey layout'a ekle


    def button_calistir_command(self): # Çalıştır butonuna tıklandığında çalışacak fonksiyon
        keyword = self.entry_keyword.text()  # Anahtar kelimeyi al
        link_count = self.spinbox_makale_sayisi.value()  # Makale sayısını al
        benzerlik_esigi = self.slider_esik.value() / 100  # Benzerlik eşiğini al

        secilen_algoritmalar = []  # Seçilen algoritmaları saklamak için liste
        if self.checkbox_bertsum.isChecked():  # BERTSUM seçiliyse
            secilen_algoritmalar.append("BERTSUM") # Listeye ekle
        if self.checkbox_textrank.isChecked():   # TEXTRANK seçiliyse
            secilen_algoritmalar.append("TEXTRANK")   # Listeye ekle
        if self.checkbox_lexrank.isChecked():    # LEXRANK seçiliyse
            secilen_algoritmalar.append("LEXRANK")  # Listeye ekle

        self.worker_thread = WorkerThread(  # İş parçacığı oluştur
            keyword,  # Anahtar kelime
            link_count,  # Makale sayısı
            benzerlik_esigi,  # Benzerlik eşiği
            secilen_algoritmalar,  # Seçilen algoritmalar
            self.text_referans.toPlainText()   # Referans metin
        )
        self.worker_thread.progress_updated.connect(self.update_progress) # İlerleme sinyali için bağlantı kur
        self.worker_thread.results_ready.connect(self.show_results) # Sonuç sinyali için bağlantı kur
        self.worker_thread.error_occurred.connect(self.show_error)   # Hata sinyali için bağlantı kur
        self.worker_thread.start()  # İş parçacığını başlat


    def update_progress(self, value): # İlerleme çubuğunu güncelleme fonksiyonu
        self.progress_bar.setValue(value)  # Değeri ayarla


    def show_results(self, results): # Sonuçları gösterme fonksiyonu
        self.text_sonuc.append(results)  # Sonuçları metin düzenleyiciye ekle


    def show_error(self, error_message): # Hata mesajını gösterme fonksiyonu
        QMessageBox.critical(self, "Hata", error_message)   # Hata mesajı kutusu göster


    def esik_degerini_guncelle(self): # Benzerlik eşiği etiketini güncelleme fonksiyonu
        deger = self.slider_esik.value()  # Slider değerini al
        self.esik_degeri_label.setText(str(deger))   # Etikete yaz


if __name__ == "__main__":  # Ana program
    app = QApplication([])  # Uygulama oluştur
    pencere = KaynakBulmaArac()  # Ana pencere oluştur
    pencere.show()  # Pencereyi göster
    app.exec_()  # Uygulamayı başlat
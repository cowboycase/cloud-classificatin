# Akbank Derin Öğrenme Bootcamp: Bulut Sınıflandırma Projesi
Bu repository, Akbank & Global AI Hub iş birliğiyle düzenlenen Derin Öğrenme Bootcamp'i kapsamında geliştirilen Bulut Sınıflandırma Projesi'ni içermektedir.

### 📝 Projenin Amacı
Projenin temel amacı, bir görüntü veri setini kullanarak, Evrişimli Sinir Ağları (CNN) mimarisi ile uçtan uca bir derin öğrenme projesi geliştirmektir. Bu süreç, veri ön işleme, model oluşturma, hiperparametre optimizasyonu, model değerlendirme ve sonuçların yorumlanması adımlarını kapsamaktadır.

### 📊 Veri Seti Hakkında Bilgi
Projede, Kaggle üzerinde halka açık olarak bulunan **"Clouds Photos"** veri seti kullanılmıştır.
*   **İçerik:** Veri seti, 7 farklı bulut türüne ait görsellerden oluşmaktadır (Cirrus, Stratus, Cumulus vb.).
*   **Yapı:** Veri seti, model eğitimi için `clouds_train` ve `clouds_test` olarak iki ana klasöre ayrılmış şekilde sunulmaktadır.
*   **Veri Seti Linki:** [Kagle - Clouds Photos](https://www.kaggle.com/datasets/fatemehmehrparvar/clouds-photos)

### 📌Kullanılan Yöntemler
Proje boyunca aşağıdaki yöntemler ve teknolojiler uygulanmıştır:
*   **Veri Ön İşleme ve Zenginleştirme (Data Augmentation):**
    *   Görüntü pikselleri `ImageDataGenerator` kullanılarak 0-1 aralığına normalize edilmiştir.
    *   Modelin genelleme yeteneğini artırmak ve ezberlemeyi (overfitting) önlemek amacıyla eğitim verilerine döndürme (rotation), kaydırma (shift), yakınlaştırma (zoom) ve yatay çevirme (horizontal flip) gibi veri zenginleştirme teknikleri uygulanmıştır.

*   **CNN Model Mimarisi:**
    *   TensorFlow/Keras kütüphanesi kullanılarak bir CNN modeli sıfırdan oluşturulmuştur.
    *   Model; `Conv2D`, `MaxPooling2D`, `Dropout` ve `Dense` katmanları gibi temel ve gerekli tüm bileşenleri içermektedir.

*   **Hiperparametre Optimizasyonu:**
    *   Modelin performansını iyileştirmek amacıyla, `dropout oranı` ve `optimizer seçimi` gibi kritik hiperparametreler üzerinde manuel denemeler yapılmıştır.
    *   En iyi performansı veren parametreler (`dropout=0.5`, `optimizer='adam'`) seçilerek nihai model (final model) bu parametrelerle eğitilmiştir.

*   **Model Değerlendirmesi:**
    *   Modelin öğrenme süreci, epoch bazında **Doğruluk (Accuracy)** ve **Kayıp (Loss)** grafikleri ile takip edilmiştir.
    *   Sınıf bazındaki performans, **Sınıflandırma Raporu (Classification Report)** ve **Karmaşıklık Matrisi (Confusion Matrix)** ile detaylı olarak analiz edilmiştir.
    *   Modelin karar mekanizmasını yorumlamak için **Grad-CAM** tekniği ile ısı haritaları oluşturulmuştur.


### 📊 Elde Edilen Sonuçlar

Yapılan optimizasyon süreci sonucunda oluşturulan final modeli, test seti üzerinde **yaklaşık %77'lik bir doğruluk (accuracy) skoruna** ulaşmıştır. Bu skor, ilk bakışta mütevazı görünse de, projenin zorlukları ve elde edilen değerli içgörüler ışığında önemli bir başarıdır. Ana bulgularımız şu şekildedir:

*   **Overfitting Kontrolü:** `EarlyStopping` mekanizması sayesinde model, ezberlemeye (overfitting) başlamadan en uygun noktada durdurularak genelleme yeteneği başarılı bir şekilde korunmuştur.

*   **Sınıf Bazında Analiz:** Karmaşıklık Matrisi, modelin görsel olarak belirgin bulut tiplerini daha iyi sınıflandırdığını; ancak **Altostratus** ve **Nimbostratus** gibi birbirine benzeyen sınıfları daha çok karıştırdığını göstermiştir. Bu, skorun sınırlı kalmasındaki ana etkenlerden biridir.

*   **Yorumlanabilirlik:** Grad-CAM görselleri, modelin tahmin yaparken rastgele değil, bulutların **dokusal ve şekilsel özelliklerine** odaklandığını ve mantıklı kararlar verdiğini doğrulamıştır.

  ### Sonuç ve Gelecek Çalışmalar
Bu proje, bir derin öğrenme modelinin durağan bir görüntüdeki bulut türünü nasıl sınıflandırabileceğini başarıyla göstermiştir. Bu, portfolyomun statik bir parçası olmanın ötesinde, sürekli geliştirilebilecek canlı bir projedir. Gelecekteki çalışmalar için vizyonum, bu temel sınıflandırma modelini, hava durumu gibi doğası gereği kaotik sistemleri anlayan ve tahminleyen daha gelişmiş bir araca dönüştürmektir:

*   **Sınıflandırmadan Kısa Vadeli Tahmine Geçiş:** Projenin en doğal bir sonraki adımı, mevcut CNN modelini bir video akışından (yer kameraları, drone'lar veya uydu görüntüleri) gelen verileri işleyebilen bir yapıya entegre etmektir. Bir **CNN-LSTM hibrit mimarisi** kullanarak, bulutların zaman içindeki morfolojik değişimini (büyüme, hareket yönü, şekil değiştirme) analiz edebiliriz. Bu, "Önümüzdeki 30 dakika içinde yağmur veya fırtına olasılığı nedir?" gibi kısa vadeli ve lokasyon bazlı tahminler yapma potansiyeli sunar.

*   **Ekstrem Hava Olaylarının Görsel İmzalarını Tespit Etme:** Hava durumu, başlangıç koşullarına son derece hassas olan kaotik bir sistemdir. Gelecekteki bir model, sadece bulut türünü etiketlemekle kalmayıp, aynı zamanda fırtına veya dolu gibi ekstrem olaylara yol açan belirli görsel desenleri ve öncü belirtileri tanımak üzere eğitilebilir. Örneğin, bir Cumulonimbus bulutunun yapısındaki belirli dönme hareketleri, bir hortumun habercisi olabilir. Bu, teorik bir konsepti, potansiyel olarak hayat kurtarabilecek pratik bir erken uyarı sistemine dönüştürme vizyonudur.


### Kaggle Notebook Linki
Projenin tüm kodlarını, detaylı teknik açıklamalarını, analizlerini ve çıktılarını içeren Kaggle Notebook'una aşağıdaki linkten ulaşabilirsiniz:

**[Projenin Kaggle Notebook'una Buradan Ulaşabilirsiniz](https://www.kaggle.com/code/cowboycase/cloud-classification)** 

import cv2
import mediapipe
import pyttsx3


camera = cv2.VideoCapture(0)  # tek kamera olduğu için 0 ı atayarak kameramızı seçiyoruz

engine =pyttsx3.init()#yazılı metni konuşmaya dönüştürür

# Kamera çözünürlüğünü ayarla
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # Pencereyi normal modda açar ve boyutlandırmaya izin verir


mpHands=mediapipe.solutions.hands #mediapipe kütüphanesini kullanarak elleri algılayan bir nesne oluşturur

hands=mpHands.Hands()

mpDraw=mediapipe.solutions.drawing_utils #elimizdeki noktaları kameraya bastırmamız gerekiyor bu değişken resme noktaları çizecek

checkThumbsUp=False

while True:

    success , img = camera.read() #boolean ve yakalanan fream(img) eğer kamera hiç bir şey yakalamazsa success false dönüyor img boş resim olarak gönderiliyor diğer türlüsü img içinde resim oluyor ve bunu ekranda görmek istiyoruz

    img = cv2.flip(img, 1)#kameranın simetriği

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hlms = hands.process(imgRGB)#elimizdeki görüntüyü renkli ele alır

    height,width,channel=img.shape #shape 3 değişken dönüdürüyor (satır sayısı(uzunluk),sütün sayısı(genişlik),rsmin kanalı)

#elimizdeki 21 noktayı ekranda görmek için
    if hlms.multi_hand_landmarks:
        #print(len(hlms.multi_hand_landmarks))#kaç tane el olduğunu gösteriyor
        for handlandmarks in hlms.multi_hand_landmarks : #noktaları ekranda görmek

            for fingerNum,landmark in enumerate(handlandmarks.landmark):
                positionX, positionY=int(landmark.x*width), int(landmark.y*height)#her bir parmak için pozisyon hesaplaması burda gerçekleşiyor

                if fingerNum>4 and landmark.y <handlandmarks.landmark[2].y :
                    break

                if fingerNum==20 and landmark.y>handlandmarks.landmark[2].y:
                    checkThumbsUp=True

            mpDraw.draw_landmarks(img,handlandmarks,mpHands.HAND_CONNECTIONS)#elimizdeki her bir elin tespit edilen noktalarını ve bu noktalar arasındaki bağlantıları çizer.

    cv2.imshow("Camera", img)#videoyu göstermek istediğimiz pencerenin ismi(Camera),yakalanan resim(img)

    if checkThumbsUp:
        engine.say("Thumbs UP!")
        engine.runAndWait()
        break

    #cv2.waitKey(1)#32 bitlik bir int döndürüyor 8 bit lazım aşağıda 8 bit yaptık
    if cv2.waitKey(1) & 0xFF==ord("q"): #q ile kapatırız .eğer karşılaştırma true dönerse
        break #yapılacak olan uygulamayı sonlandırmak


#buraya kadar görüntüyü yakalama işlemi yapıldı şimdi yapmaız gerekn yakaladığımız görüntüyü mediapipeye göndermek bundan çnce yapmamız gereken yakalanan görüntüyü rgb formatına çevirmek
#çünkü mediapipe rdj kabul ediyor bizim görselin formatı djr

#baş parmağı algılaması lazım bunun için görselde görüldüğü gibi baş parmağın ucundaki 4 numaralı parmağın y eksenindeki değerinin diğer hepsinden büyük olması lazım


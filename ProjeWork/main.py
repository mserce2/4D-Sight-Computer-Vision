import cv2
import os

per=25 #en iyi eşleşme sayısı

imgQ=cv2.imread("StarMap.png") #Yıldız resmimizi okuyoruz
h,w,c=imgQ.shape #resmimizin  şeklini yükseklik genişlik ve merkez noktası olarak değerleri değişkenlere atıyoruz
imgQ=cv2.resize(imgQ,(w//3,h//3)) #Resmimizi orantılı bir şekilde kaydetmek için size'nı ayarlıyoruz Ana resmimizin size comment ettik çünkü resimlerde sıkışma oluyor

orb=cv2.ORB_create(1000) #feature sayısını belirledik resim içersinde 1000 tane özellik arayacak
kp1,des1=orb.detectAndCompute(imgQ,None) #KeyPoints1 Description1=> anahtar noktaları bul ve tanıyıcılar ile eşleştir
# impKp1=cv2.drawKeypoints(imgQ,kp1,None) #Bulduğumuz anahtar noktalarını çizdiriyoruz.Test ettikten sonra comment yapabilirz
# cv2.imshow("KeyPointsQuery",impKp1)


path="Map"
myPicList=os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img=cv2.imread(path+"/"+y) #klasördeki resimleri okuyoruz
    #img=cv2.resize(img,(w//3,h//3)) #size işlemleri
    # cv2.imshow(y,img) #Test ettikden sonra comment yapıyoruz
    kp2,des2=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING) #norm hamming eşleşmeleri altını çizer
    matches=bf.match(des2,des1) #İlk tanımlayıcımız ile daha sonra eklediğimiz resimlerin tanımlayıcılarını karşılaştırıyoruz eşleşenler bize lazım olacak çünkü
    matches.sort(key=lambda x:x.distance)  #eşleşmelerin mesafesine bakıyoruz eşleşme mesafesi ne kadar düşük olursa o kadar iyi eşleşme olur
    good=matches[:int(len(matches)*(per/100))] #iyi olan eşleşmelerin yüzdesini alıyoruz.Per =25 olduğu için en iyi 25 eşeleşme için bunu yapacağız
    imgMatch=cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2) #Son resimlerimizin=>img,Anahtar Noktaları =>kp2, İlk Hedef resim =>imgQ anahtar noktaları =>Kp1, Ortak iyi olan eşleşmeler =>Good hepsini çizdir good[:100] en iyi 100 eşleşmeyi verir :20 derseniz en iyi 20 eşleşmeyi getirir
    #NOT:en iyi 100 eşleşmeyi çağırma sebebim algoritmanın dinamikliğini göstermek.Ve zaten toplamda 5 eşleşme bulunuyor :)
    #Ancak cv2.ORB_create(9000) sayısını artırırsam daha fazla eşleşen nokta bulacaktır.Bunun resmini depoda bulabilirsiniz.
    imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    cv2.imshow(y,imgMatch)



cv2.imshow("Original Image",imgQ)
cv2.waitKey()





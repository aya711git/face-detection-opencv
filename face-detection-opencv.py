# استيراد مكتبة OpenCV ومكتبات مساعدة
import cv2
import os
from google.colab.patches import cv2_imshow  # لعرض الصور في Google Colab

# الحصول على المسار الخاص بملف Haar Cascade المدمج مع OpenCV
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# تحميل الصورة من المسار المحدد
image = cv2.imread("/content/drive/MyDrive/face.jpg")

# تحويل الصورة إلى التدرج الرمادي (لأن كاشف الوجوه يعمل بشكل أفضل على صور رمادية)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تحميل نموذج Haar Cascade الخاص بالكشف عن الوجوه
face_cascade = cv2.CascadeClassifier(cascade_path)

# التأكد من تحميل ملف Haar Cascade بنجاح
if face_cascade.empty():
    raise Exception("خطأ في تحميل ملف Haar Cascade.")

# استخدام النموذج لاكتشاف الوجوه داخل الصورة
# المعامل الأول: الصورة الرمادية
# المعامل الثاني: scaleFactor (لتقليص حجم الصورة تدريجياً)
# المعامل الثالث: minNeighbors (عدد الجيران المطلوب لتأكيد وجود وجه)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# رسم مستطيل حول كل وجه يتم اكتشافه
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # (255,0,0) اللون أزرق، سمك الخط 2

# عرض الصورة النهائية مع تحديد الوجوه
cv2_imshow(image)

# الانتظار لالتقاط أي مفتاح (لن يعمل في Colab لكنه يستخدم عادة في بيئة سطح المكتب)
cv2.waitKey(0)

# إغلاق جميع النوافذ المفتوحة (في حال تشغيل الكود على جهاز محلي)
cv2.destroyAllWindows()

import numpy as np
import glob
from sklearn.cluster import DBSCAN,KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.pyplot as plt

# #############################################################################
# Compute DBSCAN
def mesr(image):
  step=40
  # gray=cv2.imread(image,0)
  gray=cv2.resize(image,(512,512))
  _,last_img=cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
  total_img=np.zeros((512,512),dtype=np.uint8)
  img=last_img
  teml_img=np.ones((512,512))*255
  last_teml=np.uint8(teml_img)
  for threshold in range(0,180,step):#hyper 180
      last_img=img
      _,img=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
      
      teml_img=cv2.bitwise_xor(img,last_img)
      teml_img2=cv2.dilate(teml_img,np.array([[1,1,1],[1,1,1],[1,1,1]]),iterations=1)
      residual=cv2.bitwise_and(teml_img2,last_teml)
      last_teml=teml_img
      residual=cv2.dilate(residual,np.array([[1,1,1],[1,1,1],[1,1,1]]),iterations=1)
      residual=cv2.bitwise_and(teml_img,residual)
      total_img=total_img+residual
  return total_img
img_list = glob.glob('F:/project_2/New_folder/data/downloads/*.jpg')
for img_address in img_list:
  img= cv2.imread(img_address,0)
  # img= cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
  # db = DBSCAN(eps=0.3, min_samples=10).fit(X)
  # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  # core_samples_mask[db.core_sample_indices_] = True
  # labels = db.labels_
  img= mesr(img)
  img_posision= (img==255)
  db = DBSCAN(eps=5, min_samples=3).fit(np.argwhere(img_posision))
  img_out= np.zeros_like(img)
  cluster_label=db.labels_
  hist=np.histogram(cluster_label,bins=np.unique(cluster_label))
  kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array([hist[0],np.zeros_like(hist[0])]).transpose())
  cluster=kmeans.labels_
  plt.plot(list(range(len(hist[0]))),hist[0])
  plt.show()
  posision_out =np.argwhere(img_posision)[cluster_label>5].transpose()
  print(posision_out[:,:10])
  img_out[posision_out[0],posision_out[1]]= 255
  plt.imshow(img)
  plt.show()
  plt.imshow(img_out)
  plt.show()
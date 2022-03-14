import cv2 
import ctypes
   
## get Screen Size
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def showImagec(filename):
     W,H = screensize
     W=W*0.9
     H=H*0.9
     oriimg = filename
     height, width, prof = oriimg.shape

     scaleWidth = float(W)/float(width)         #rescaling the Width
     scaleHeight = float(H)/float(height)       #rescaling the height

     if scaleHeight>scaleWidth:                 #case if the width is bigger            
          imgScale = scaleWidth
     else:                                      #case if the height is bigger
          imgScale = scaleHeight

     newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale 
     newimg = cv2.resize(oriimg,(int(newX),int(newY)))              #rescaling the image

     return newimg, imgScale
if __name__ == '__main__':
     filename = 'C:/Users/antoi/Pictures/digiCamControl/Session1/Bas.jpg'
     showImagec(filename)       
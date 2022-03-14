# import the necessary packages
import cv2 as cv                     
from matplotlib import pyplot as plt 
import pandas as pd                 
from scipy import ndimage            
from skimage import color            
from skimage.measure import regionprops_table  
from math import sqrt                
import numpy as np
from showImage import showImage
from showImagec import showImagec
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations
import keyboard
#Step 1 CROPPING
refPt = []                           #Initializing the table for the coordinates of the cropped region
cropping = False                     #boolean that indicate us if we are in cropping mode or not
source='C:/Users/antoi/Pictures/T17/DSC_0350.jpg'

def click_and_crop(event, x, y, flags, param): #function for cropping
	global refPt, cropping                     # grab references to the global variables
	if event == cv.EVENT_LBUTTONDOWN:          # if the left mouse button was clicked, record the starting
		refPt = [(x, y)]                       # (x, y) coordinatems and indicate that cropping is being performed
		cropping = True
	elif event == cv.EVENT_LBUTTONUP:          # check to see if the left mouse button was released
		refPt.append((x, y))                   # record the ending (x, y) coordinates and indicate that
		cropping = False                       # the cropping operation is finished
		cv.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2) # draw a rectangle around the region of interest
		cv.imshow("image", image)

#We are going to switch from an image to another, so we copy all the image 
image = cv.imread(source,0) 
image_couleur=cv.imread(source)
CloneImageCouleur=image_couleur.copy()
copie_image = image.copy() 
image, coef_Zoom=showImage(image)
copie_image_couleur = image.copy()
coef_Zoom=1/coef_Zoom                                                      
cv.namedWindow("image")
cv.setMouseCallback("image", click_and_crop) 

#We cropped the image at the good size to remove the scale and borders,
#or to work on a specific flake

while True:                    
	cv.imshow("image", image)  
	key = cv.waitKey(1) & 0xFF
	if key == ord("r"):        
		image = copie_image_couleur.copy()  
	elif key == ord("c"):
		break

if len(refPt) == 2:  
    a1=round(refPt[0][1]);a2=round(refPt[1][1]);a3=round(refPt[0][0]);a4=round(refPt[1][0])
    a5=round(refPt[0][1]*coef_Zoom);a6=round(refPt[1][1]*coef_Zoom);a7=round(refPt[0][0]*coef_Zoom);a8=round(refPt[1][0]*coef_Zoom)
    roi = copie_image_couleur[a1:a2, a3:a4]  
    roi_grand = copie_image[a5:a6, a7:a8]
    roi_couleur=CloneImageCouleur[a5:a6, a7:a8]
    cv.imshow("ROI", roi)
    cv.waitKey(0)# close all open windows

#Setting the scale. We draw a line one the physical scale that mesure 150mm to find
#the pixel to mm coefficient

cropping = False
def scale(event, x, y, flags, param): 
	global refPt, cropping # grab references to the global variables
	if event == cv.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	elif event == cv.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv.line(image, refPt[0], refPt[1], (0, 255, 0), 2) #draw the line for the scale
image = cv.imread('C:/Users/antoi/Pictures/T17/DSC_0350.jpg',0)
copie_image_echelle = image.copy()
image, inu=showImage(image)
copie_image_couleur = image.copy()
cv.namedWindow("scale")
cv.setMouseCallback("scale", scale) 

while True:
	cv.imshow("scale", image)
	key = cv.waitKey(1) & 0xFF
	if key == ord("r"):
		image = copie_image_echelle.copy()
	elif key == ord("c"):
		break
p0=round(refPt[0][0]);p1=round(refPt[0][1]);p2=round(refPt[1][0]);p3=round(refPt[1][1]) 
pixeltomm=150/(sqrt((p0-p2)**2+(p1-p3)**2)*coef_Zoom) #The scale here is 20mm, so we can determine the conversion between pixel and mm
cv.destroyAllWindows()
#pixeltomm=0.12032049421456982
#Applying filter to reduce noise suche as small holes.
#sharpening the edge of th flakes

kernel = np.array([[0, -1, 0],
                  [-1, 5,-1],
                  [0, -1, 0]]) 
roig1 = cv.filter2D(src=roi_grand, ddepth=-1, kernel=kernel)

#threshold using otsu function
#plt.hist(roi_grand.flat , bins=100, range=(0,255))
ret, thresh = cv.threshold(roi_grand, 141, 255, cv.THRESH_BINARY)
#ret, thresh = cv.threshold(roi_grand, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

#invert black and white

thresh[thresh==255]=254                                                 
thresh[thresh==0]=255
thresh[thresh==254]=0
thresh= cv.bilateralFilter(thresh,5,70,70)

#Working on the image to make it cleaner with less noise and less chances of 
#image processing failsdd

kernel = np.ones((3,3),np.uint8) 
dilated = cv.dilate(thresh,kernel,iterations=4)
eroded = cv.erode(dilated,kernel,iterations=3) 

mask = eroded ==255 #converting in a boolean table 

#Detecting and labeling all the flakes

s = [[1,1,1],[1,1,1],[1,1,1]]                               
label_mask, num_labels = ndimage.label(mask,structure=s)   
#img2 = color.label2rgb(label_mask, bg_label=0)
cv.destroyAllWindows()

#Seraching for geometrical caracteristics of each flakes
#such as area, boudning box and feret diameter


propList=['area','label','bbox','feret_diameter_max','orientation',
            'major_axis_length',
            'minor_axis_length',
            'mean_intensity']                             #choosing the parameter i want to find with the regionprop function
props =   regionprops_table(label_mask,roi_grand,properties=propList) #using the regionprops_table
Data=pd.DataFrame(props)

#Converting all the deter in mm and exporting it to an excel

def pixelto_mmsq(x):
	return x*(pixeltomm)**2
def pixelto_mmscale(x):
	return x*(pixeltomm)
Data['area'] = Data['area'].apply(pixelto_mmsq)     
Data['feret_diameter_max'] = Data['feret_diameter_max'].apply(pixelto_mmscale)             #converting the area in pixel² to mm²                                          #creating an excel file with all the datas

#We are now looking for detecting the edges of each flakes, find the orientation of it
#and calculating the length in the direction of the fiber and the witdh

"""
How it works :
    Each flakes has his label (number) and a bounding box that encapsulate 
    the entire flake in it,on the label_mask image.
    We isolate each flakes with his bounding box
    Find the edges
    Apply the ApproxPolyDP function
    Calculate the coordinates of all points of the polygon
    Find the parallel line
    Calculate all the geomtetrical feature we want to know
    
    If we find to set of edge that a both parallel, the code ask the user to chose the 
    principal direction of the fiber
    If there are more or less than 4 edges, it asks us to parameter the sensibility
    of the approxPolyDP function
    
"""



allong=  [];lmoy = [];lar=[];lmin = [];lmax= []; dmax=[]; rep=[]; epsi=[]; elmin=[]; elmaj=[]; aire=[]
allo=0  ; dtot=0                  
lx,ly=label_mask.shape
dima=np.zeros((num_labels,2))                               
print(num_labels)
hauteur_roi_grand,largeur_roi_grand = roi_grand.shape
dtot=0
#we isolate each flakes that has his own number (through labelling) and make a image with only this flakes out of it
for k in range(num_labels):
    Delete=False
    copie_label_mask=label_mask.copy()  ;par=[];    par2=[]
    print(k) 
    y=Data['bbox-0'][k] ;ym=Data['bbox-2'][k];x=Data['bbox-1'][k];xm=Data['bbox-3'][k]
    ar=Data['area'][k]                   
    copie_roi_grand=roi_grand.copy()
    if y<70:
        y=0;ym=round(ym)+70;
    elif ym>hauteur_roi_grand-70:
        ym=hauteur_roi_grand; y=round(y)-70;
    else:
         ym=round(ym)+70;y=round(y)-70 
    if x<70:
        x=0;xm=round(xm)+70
    elif xm>largeur_roi_grand-70:
        xm=largeur_roi_grand; x =round(x)-70; 
    else:
        x =round(x)-70; xm=round(xm)+70
    box=copie_roi_grand[y:ym,x:xm]
    box_noir_blanc=copie_label_mask[y:ym,x:xm]
    box_couleur=roi_couleur[y:ym,x:xm]
    lab=Data['label'][k]
    boxg=np.zeros(box_noir_blanc.shape)
    boxg[box_noir_blanc!=lab]=120
    kernel = np.ones((3,3),np.uint8) 
    boxg = cv.dilate(boxg,kernel,iterations=4)
    box[boxg==120]=0
    ret3, thresh3 = cv.threshold(box, Data['mean_intensity'][k]+10, 255, cv.THRESH_BINARY)
    thresh3= cv.GaussianBlur(thresh3, (3, 3), 0)
    sobelx = cv.Sobel(thresh3,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(thresh3,cv.CV_64F,0,1,ksize=5)
    sx=np.sum(abs(sobelx))
    sy=np.sum(abs(sobely))
    if sx>=sy:
        pc=(1,1000)
    else:
        pc=(1000,1)
    pc=(1,1000)    
    box_noir_blanc[box_noir_blanc!=lab]=0
    box_noir_blanc[box_noir_blanc==lab]=255
    box_noir_blanc=np.uint8(box_noir_blanc)
    contours1,hierarchy1 = cv.findContours(box_noir_blanc, 1, cv.CHAIN_APPROX_NONE)
    cnt1 = contours1[0]
    sortie= False
    ep = 0.025
    contours3,hierarchy3 = cv.findContours(box_noir_blanc, 1, 2)      
    cnt3 = contours3[0]                                       
    rect3 = cv.minAreaRect(cnt3)
    v1,v2=box_noir_blanc.shape                              
    box3 = cv.boxPoints(rect3)                                
    box3 = np.int0(box3)
    box2=box3.copy()
    for i in range(len(box2)):
        box2[i][1]=v1-box2[i][1]
    h1 = box2[0][0];h2 = box2[0][1]; h3 = box2[1][0];h4 = box2[1][1]
    h5 = box2[2][0];h6 = box2[2][1]; h7 = box2[3][0];h8 = box2[3][1]
    h1b = box3[0][0];h2b = box3[0][1]; h3b = box3[1][0];h4b = box3[1][1]
    h5b = box3[2][0];h6b = box3[2][1]; h7b = box3[3][0];h8b = box3[3][1]
    line_thickness = 2

    if h2==h4 :
        h2=h2+1
    elif h1==h3 : 
        h1=h1+1
    if h6==h8 :
        h6=h6+1
    elif h5==h7 :
        h5=h5+1
    hel=False
    passa=False
    while sortie ==False:
        clonebox=box_couleur.copy()
        sortiek =False
        epsilon1 = ep*cv.arcLength(cnt1,True)
        approx1 = cv.approxPolyDP(cnt1,epsilon1,True)
        if (len(approx1)==4 or len(approx1)==5)  and passa==False:
            break
        for w in range(len(approx1)-1):
                    r0=w; r1=w+1
                    c1,c2 = approx1[r0][0]; c3,c4 = approx1[r1][0]
                    cv.line(clonebox, (c1, c2), (c3, c4), 255, 2)    
        c1,c2 = approx1[len(approx1)-1][0]; c3,c4 = approx1[0][0]
        cv.line(clonebox, (c1, c2), (c3, c4), 255, 2)
        clonebox, coef_Zoom =showImagec(clonebox)
        cv.imshow("Plus precis= m | Moins preecis = p | exit = e | continue = * | blur = f",clonebox)
        while sortiek==False:
            cv.waitKey(1000)
            if keyboard.read_key() == "m":
                ep=ep-0.005
                sortiek=True
            elif keyboard.read_key() == "p":
                ep=ep+0.005
                sortiek=True
            elif keyboard.read_key() == "f":
                box_noir_blanc = cv.medianBlur(box_noir_blanc,7)
                contours1,hierarchy1 = cv.findContours(box_noir_blanc, 1, cv.CHAIN_APPROX_NONE)
                cnt1 = contours1[0]
                sortiek=True
            elif keyboard.read_key() == "h":
                hel=True
                sortiek=True
                sortie=True
            elif keyboard.read_key() == "d":
                sortiek=True
                sortie=True                
            elif keyboard.read_key() == "*":
                sortiek=True
                sortie=True
                Delete=True
        passa=True
        cv.destroyAllWindows()
        cv.waitKey(100)
    cv.waitKey(100)
    if Delete==True:
        continue
    for i in range(len(approx1)):
        approx1[i][0][1]=v1-approx1[i][0][1]
    for i in range (0,len(approx1)-1):
        c1,c2 = approx1[i][0]; c3,c4=approx1[i+1][0];
        if c1>=c3:
            c5= c1-c3 ; c6=c2-c4
        else :
            c5= c3-c1 ; c6=c4-c2
        par2.append((c5,c6))
    c1,c2 = approx1[len(approx1)-1][0]; c3,c4=approx1[0][0];
    if c1>=c3:
        c5= c1-c3 ; c6=c2-c4
    else :
        c5= c3-c1 ; c6=c4-c2
    par2.append((c5,c6))
    for i in range (0,len(approx1)-1):
        c1,c2 = par2[i]
        if c1==0:
            c1=1
        coef=np.arctan(c2/c1)
        if coef<-1.3:
            coef=coef+3.14159
        par.append(coef)
    c1,c2 = par2[len(par2)-1]
    if c1==0:
        c1=1
    coef=np.arctan(c2/c1)
    if coef<-1.3:
            coef=coef+3.14159
    par.append(coef)
    com=list(range(0,len(par)))
    com2=[i for i in combinations(com,2)]
    com3=[]
    for o in range((len(com2))):
        i1,i2=com2[o]
        i4=len(approx1)-1
        i3=abs(i1-i2)
        if i3!=1  and i3!=i4:
            com3.append(com2[o])
    det2=10
    opr=10
    rdt2=20
    compte=0
    nex=False
    for o in range((len(com3))):
            k1,k2=com3[o]
            s1=par[k1]; s2=par[k2];
            rdt=abs(s1-s2)
        
            if rdt<0.07:
                compte=compte+1
            if rdt<=rdt2:
                rdt2=rdt
                m1,m2=k1,k2
    if nex==True:
        continue
    if hel==True:
        compte=len(com3)
            
    if compte>=2 or compte==len(com3):
        rdt2=20
        sortieb=False
        clonxc=box_couleur.copy()
        cv.line(clonxc, (h1b, h2b), (h3b, h4b), (0, 255, 0), thickness=line_thickness)
        cv.line(clonxc, (h3b, h4b), (h5b, h6b), (0,0,255), thickness=line_thickness)    
        if h1>=h3:
            h9= h1-h3 ; h10=h2-h4
        else :
            h9= h3-h1 ; h10=h4-h2
        if h3>=h5:
            h11= h3-h5 ; h12=h4-h6
        else :
            h11= h5-h3 ; h12=h6-h4
        apc=np.arctan([pc[1]/pc[0]])
        if apc<-1.3:
            apc=apc+3.14159
        if h9==0:
            h9=1
        if h11==0:
            h11=1
        ar1=abs(np.arctan(h10/h9))
        ar2=abs(np.arctan(h12/h11))
        af1=abs(apc-ar1)
        af2=abs(apc-ar2)
        if af1<=af2:
            f10=h10;f9=h9
        else:
            f10=h12;f9=h11
        if f9==0:
            f9=1
        par3=[]
        for o in range((len(com3))):
            k1,k2=com3[o]
            s1,s2=par2[k1]; s3,s4=par2[k2];
            if s1==0:
                s1=1
            elif s3==0:
                s3=1
            ag1=np.arctan(s2/s1) ; ag2=np.arctan(s4/s3);ag3=np.arctan(f10/f9)
            #Gerer les problèeme d'abcsisse et d'ordonnée similaire mais de coté axe différents
            if ag1<-1.3:
                ag1=ag1+3.14159
            if ag2<-1.3:
                ag2=ag2+3.14159
            if ag3<-1.3:
                ag3=ag3+3.14159
            
            rdt1=abs(ag1-ag3)+abs(ag2-ag3)

            if rdt1<=rdt2:
                rdt2=rdt1
                m1,m2=k1,k2
            par3.append((rdt1,o))
        par3.sort()
    if nex==True:
        continue
    n1=m1; n2=m1+1 ; n3=m2; n4=m2+1     
    if n4 == len(approx1):
        n4=0
    elif n2 == len(approx1):
        n2=0
    if approx1[n1][0][0]-approx1[n2][0][0]==0:
         approx1[n1][0][0]= approx1[n1][0][0]+1
        
    a=(approx1[n1][0][1]-approx1[n2][0][1])/(approx1[n1][0][0]-approx1[n2][0][0]);
    b=approx1[n1][0][1]-approx1[n1][0][0]*a
    d=abs((a*approx1[n3][0][0]-approx1[n3][0][1]+b))/(sqrt(a**2+1))
    o1,o2=approx1[n1][0] ;o3,o4=approx1[n2][0]
    l1=sqrt((o1-o3)**2+(o2-o4)**2)
    o1,o2=approx1[n3][0] ;o3,o4=approx1[n4][0]
    l2=sqrt((o1-o3)**2+(o2-o4)**2)
    lmo=(l1+l2)/2
    al=lmo/d
    if l1<=l2 : 
            lmaxi=l2;lmini=l1
    else :
            lmaxi=l1;lmini=l2     
    dmm= pixelto_mmscale(d)
    pas= dmm/2
    fid=lmaxi-lmini
    aug=fid/pas
    long=lmini
    ps=0
    while ps<=dmm :
        lot=long
        rep.append(lot)
        long=long+aug
        ps=ps+2 
        dtot=dtot+2
    lar.append(d);lmoy.append(lmo);lmax.append(lmaxi);lmin.append(lmini);allong.append(al)
    approx2=approx1.copy()
    for i in range(len(approx1)):
        approx2[i][0][1]=v1-approx1[i][0][1]

    epsi.append(ep)
    aire.append(Data['area'][k])
    elmin.append(Data['minor_axis_length'][k])
    elmaj.append(Data['major_axis_length'][k])
    if k%7==0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()
            
        ax[0].imshow(box_noir_blanc, cmap=cm.gray)
        ax[0].set_title('Threshold')  
        ax[0].set_xlim((0, box_noir_blanc.shape[1]))
        ax[0].set_ylim((box_noir_blanc.shape[0], 0))
        for w in range(len(approx2)-1):
                    r0=w; r1=w+1
                    c1,c2 = approx2[r0][0]; c3,c4 = approx2[r1][0]
                    ax[1].plot((c1, c3), (c2, c4))    
        c1,c2 = approx2[len(approx2)-1][0]; c3,c4 = approx2[0][0]
        ax[1].plot((c1, c3), (c2, c4))     
        ax[1].imshow(box,'gray')
        ax[1].set_title('Fitted Polygone')  
        ax[1].set_xlim((0, box_noir_blanc.shape[1]))
        ax[1].set_ylim((box_noir_blanc.shape[0], 0))
        c1,c2 = approx2[n1][0]; c3,c4 = approx2[n2][0]
        ax[1].plot((c1, c3), (c2, c4))
        c1,c2 = approx2[n3][0]; c3,c4 = approx2[n4][0]
        ax[1].plot((c1, c3), (c2, c4))
        
        ax[2].set_title('Parallel lines')
        ax[2].set_xlim((0, box_noir_blanc.shape[1]))
        ax[2].set_ylim((box_noir_blanc.shape[0], 0))
        r0=m1; r1=m1+1
        if r1==len(approx2):
            r1=0
        c1,c2 = approx2[r0][0]; c3,c4 = approx2[r1][0]
        ax[2].imshow(box_noir_blanc*0,'gray')
        ax[2].plot((c1, c3), (c2, c4))    
        r0=m2; r1=m2+1
        if r1==len(approx2):
            r1=0
        c1,c2 = approx2[r0][0]; c3,c4 = approx2[r1][0]
        ax[2].plot((c1, c3), (c2, c4))     

donnee1 =pd.DataFrame({'aire':aire,'allongement' :allong, 'lmoy' : lmoy, 'larg' : lar, 'lmax':lmax, 'lmin':lmin, 'epsilon':epsi, 'Minor':elmin, 'Major':elmaj})
donnee1['lmoy'] = donnee1['lmoy'].apply(pixelto_mmscale) 
donnee1['larg'] = donnee1['larg'].apply(pixelto_mmscale) 
donnee1['lmax'] = donnee1['lmax'].apply(pixelto_mmscale) 
donnee1['lmin'] = donnee1['lmin'].apply(pixelto_mmscale)
donnee2=pd.DataFrame( {'repartition':rep})
donnee2['repartition'] = donnee2['repartition'].apply(pixelto_mmscale) 
donnee1.to_excel('Donnée1_17_0.xlsx', index=False)
donnee2.to_excel('Donnée2_17_0.xlsx', index=False)
Data.to_excel('Donnée3_17_0.xlsx',index=False)   

#jdddfffffpmmmmf*

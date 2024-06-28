import cv2
import numpy as np

import model as mm
import mser
import swt
import os
import uuid

# for cancatinate tow boxes
def concatinate_box(box1, box2):

    x, y, w, h = box1
    x0, y0, w0, h0 = box2

    x1 = (x, y)
    x3 = (x + w,y + h)
    y1 = ( x0,y0)
    y3 = (x0 + w0,y0 + h0)


    z1=min(x1[0],y1[0])
    z2=min(x1[1],y1[1])
    z3=max(x3[0],y3[0])-z1
    z4=max(x3[1],y3[1])-z2
    return np.array([z1,z2,z3,z4])
def show(ss,img):
    cv2.imshow( ss, img)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# get the combination of intersected boses
def group_rectangles(rectangles ):
    bboxes = rectangles
    rectangles=rectangles.tolist()

    x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    grouped_rectangles = []
    while(len(rectangles)>0):
            re=rectangles[0]
            group = re
            rectangles.remove(re)
            stop=True
            while (stop):
                  stop=False
                  for recc in rectangles:
                            x1_overlap = max(group[0],recc[0] )
                            y1_overlap = max(group[1], recc[1])
                            x2_overlap = min(group[0] + group[2],recc[0] + recc[2])
                            y2_overlap = min(group[1] + group[3],recc[1] + recc[3])
                            area = max(0, x2_overlap - x1_overlap + 1) * max(0, y2_overlap - y1_overlap + 1)
                            if area >= 5:
                                stop=True
                                group=concatinate_box(group,recc)
                                rectangles.remove(recc)
            grouped_rectangles.append(group)
    return grouped_rectangles
#cany  and mser
def getMSER(img):
    a, b, c = img.shape
    gray = swt.get_grayscale(img)
    ggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('my Image  ', img)
    mean = cv2.mean(img)
    mean = (mean[0] + mean[1] + mean[2]) / 3
    cliplimit = (255 / mean) * 1.8
    Iclahe = mser.clahe(img, clipLimit=cliplimit)
    #cv2.imshow('the  clahe image ', Iclahe )
    boxes, regions, RImage = mser.detect_mser(ggray)
    edges = mser.canny_detector(img, 30, 60)
    merged = np.zeros_like(edges)
    for i in range(a):
        for j in range(b):
            if (edges[i, j]):
                for k in range(-2, 2):
                    for l in range(-2, 2):
                        if (b > (k + j) > 0 and a > (l + i) > 0):
                            if (RImage[i + l, j + k] > 0):
                                merged[i, j] = 255
    merged = merged.astype(np.uint8)
    return merged ,gray,boxes
#swt
def get_swt(gray,img):
    gradients = swt.get_gradients(gray)
    swt0 = swt.apply_swt(img, gradients, True)
    swt0 = (255 * swt0 / (swt0.max()+1)).astype(np.uint8)
    labels, components = swt.connected_components(swt0,threshold=5)
    labels = labels.astype(np.float32) /( labels.max()+1)
    labels= (labels*255.).astype(np.uint8)
    return components,labels,
def text_swt(merged, gray):
    components, labels = get_swt(gray, merged)
    Alable0, Acompon0 = swt.variance_discard_non_text(gray, components)
    Alable0, Acompon0 = swt.filter(Alable0, Acompon0)
    #cv2.imshow(' aspect ratio  detect 00', Alable0)
    cv2.imwrite("strok.png",Alable0)
    return Alable0
def filter_swt(c,img):
    #cv2.imshow('the filter immmg  gggg', img)
    c=c.tolist()
    cc=[]
    for box in c:
        x, y, w, h = box
        ss=0
        for i in range(h):
            for j in range (w):
               if(  img[y+i,x+j]>110):
                   ss=ss+1
        if(ss*10>(w*h)) :
            cc.append(box)
    return cc
    return  result
def filter_box_by_pixel_exist(merged_bboxes,MImag):
    c=[]
    for box in merged_bboxes:
        x, y, w, h = box
        s=0
        for i in range(h):
            for j in range (w):
             if(MImag[y+i,x+j]):
                s=s+1
        if (s>0):
          c.append(box)
    c=np.array(c)
    return c
def boxes_to_image(boxes,img):
    img = cv2.copyMakeBorder(img, 5, 5, 5,5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    result=np.zeros_like(img).astype(np.uint8)
    boxes=np.array(boxes)
    for box in boxes:
        x, y, w, h = box
        for i in range(h):
            for j in range (w):
                  result[y + i, x + j] =img[y+i,x+j]
    return  result
def boxe_to_image(box, img ,hh=3,vv=3,padin=5):
    x, y, w, h = box
    x = x - hh + 5
    y = y - vv + 5
    w += 2 * hh
    h += 2 * vv
    result = np.ones(w * h * 3, dtype="uint8").reshape(h, w, 3)
    for i in range(h):
        for j in range(w):
            result[i, j] = img[y + i, x + j]
    return result
# concatinate the character to get word
def get_word(img,rect_result  ,threshod = 6,hh=(0,1000),vv=(0,1000)):
    listt=[]
    high,width, _=img.shape
    IMAG = np.ones(high*width, dtype="uint8").reshape( high,width)
    IMAG=IMAG*-1
    ind=0
    for box in rect_result :
        listt.append(box)
        x, y, w, h = box
        for i in range(h):
            for j in range(w):
                IMAG[y +i, x +j]= ind
        ind+=1
    goinlistt=[[] for i in range(len(listt))]

    for box ,indx0 in zip(rect_result,range(len(rect_result))) :
        x, y, w, h = box
        x1=(x,y)
        x2=(x+w,y)
        x3=(x+w,y+h)
        x4=(x,y+h)

        # vertical merged
        if(vv[1]>listt[indx0][0]>vv[0]):

            if(x2[0]+threshod<width ):
               previos=-2
               for k in range(x2[1],x3[1]):
                    if (IMAG[k,x2[0]+threshod ]>-1and previos !=IMAG[k,x2[0]+threshod ]):
                            previos=IMAG[k,x2[0]+threshod ]
                            indx1 =IMAG[k,x2[0]+threshod ]
                            if(alinement(listt[indx1],listt[indx0],'v')):
                                #if(0.75<listt[indx0][3]/listt[indx1][3]<1.33):
                                #print("  eeeeee==  ",indx0, "   ", indx1)
                                goinlistt[indx0].append(indx1)
        # horisontal merged
        if (hh[0]<listt[indx0][0] <hh[1]):


            if(x3[1]+threshod<high  ):
                previos = -2
                for k in range(x4[0],x3[0] ):
                    if (IMAG[ x3[1] +threshod,k]>-1 and IMAG[ x3[1] +threshod,k] !=previos):
                        previos = IMAG[ x3[1] +threshod,k]
                        indx1 = IMAG[x3[1] + threshod,k]
                        if (alinement(listt[indx1], listt[indx0], 'h')):
                            goinlistt[indx0].append(indx1)


    for i in range(len(goinlistt)):
        for j in goinlistt[i]:
            x,y,w,h = concatinate_box(listt[i], listt[j])
            listt[i][0]=x
            listt[i][1]=y
            listt[i][2]=w
            listt[i][3]=h
            my=listt[j]
            for aa in range(len(listt)):
                if (my is listt[aa]):
                    listt[aa] = listt[i]

    tupl = [tuple(arr) for arr in listt]
    unique_set = set(tupl)
    listt = [np.array(t) for t in unique_set]

    return  listt
# chek that my boxes are alinement to join it
def alinement(b1,b2,s):
    if(s=="h") :
        x1=[b1[0],b1[0]+b1[2]]
        x2=[b2[0],b2[0]+b2[2]]
    elif(s=="v"):
        x1 = [b1[1], b1[1] + b1[3]]
        x2 = [b2[1], b2[1] + b2[3]]
    else:
        return False
    if (x1[0] > x2[0]):
        x3 = x1
        x1 = x2
        x2 = x3
    if (x1[1] >= x2[1]):
        return True
    else:
        d1 = x1[1] - x2[0]
        d2 = x2[1] - x1[1]
    if((d1/d2)>0.6):
        return True
    else:
        return False
def word_region(img):
    merged,gray,merged_bboxes=getMSER(img)
    swt=text_swt(merged,gray)
    ss = np.copy(img)
    #merged_bboxes , weights  = cv2.groupRectangles(merged_bboxes, 1, 0.2)
    #swt = np.concatenate((swt, merged_bboxes))
    merged_bboxes=np.array(filter_swt(merged_bboxes,swt))
    for box  in merged_bboxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    #cv2.imshow('the result ytwrehg grouprect', ss)
    #cv2.imwrite("000.png",ss)
    #print(type(merged_bboxes))
    print(len(merged_bboxes.shape))
    if(len(merged_bboxes.shape)>=2):
        ddd=group_rectangles(merged_bboxes)
        ss = np.copy(img)
        for box  in ddd:
                cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        #cv2.imshow('the result of grouprect', ss)
        #cv2.imwrite("111.png", ss)
        word=ddd.copy()
        n=0
        nn=len(merged_bboxes)
        #get word

        while(n!=nn):
                #print("nn=================",nn)
                n=nn
                word=get_word(img,word,hh=(380,500),vv=(0,380))
                nn=len(word)




        return word
    else:
        return []
def write_on_image(img, pred):

    ss = word_region(img)
    myimg =np.copy(img)
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    i=0
    for s in ss:
        ii = boxe_to_image(s, img)
        ss=""
        if (s[2] * s[3] > 150):
            if (s[2] < s[3]):
                ii = cv2.rotate(ii, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ii=mm.my_fitting(ii)

                tex=mm.test2(pred,ii)
                ss = tex[0]
                #ss=tex[0].split("[UNK]")[0]
                #ss = ss.Replace("[UNK]", "");

                ss = ss.replace("[UNK]", "")
                ss = ss.replace("@", "/")
                ss = ss.replace("$", ".")
                cv2.imwrite("result/" + str(i) +ss+ ".png", ii)

            else:
                ii=mm.my_fitting(ii)
                #cv2.imshow("simple ", ii)
                tex=mm.test2(pred,ii)
                ss=tex[0]
                # ss = ss.Replace("[UNK]", "");

                ss = ss.replace("[UNK]", "")
                ss = ss.replace("@", "/")
                ss = ss.replace("$", ".")
                cv2.imwrite("result/" + str(i) +ss+ ".png", ii)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(myimg, ss, (s[0], s[1]), font, 0.5, (255, 20, 20), 0, cv2.LINE_AA)
            i+=1

    return myimg
def get_textimages( img):
    ss = word_region(img)

    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    i=0
    images=[]
    for s in ss:
        ii = boxe_to_image(s, img)
        if (s[2] * s[3] > 150):




            images.append(ii)
    """imgname = os.path.join("data01", f'{str(uuid.uuid1())}.png')

    cv2.imwrite(imgname, ii)"""
    i += 1
    return images
def main():
    pr=mm.get_pr("models/model05")
    link = "dedicine_label/d1.png"
    img = cv2.imread(link)
    img = cv2.resize(img, (500, 300))
    result = write_on_image(img, pr)
    show("result ", result)

"""
link="dedicine_label/d1.png"
img = cv2.imread(link)
img = cv2.resize(img, (500, 300))
result=write_on_image(img,pr)
show("result ",result)

"""
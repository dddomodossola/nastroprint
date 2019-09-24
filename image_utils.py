import cv2
import numpy as np
import shutil #to move files
import os

def stitch(imgTop, imgBottom, percent_overlap=50.0, matching_method=cv2.TM_SQDIFF_NORMED):
    """
    percent_overlap = minima percentuale di sovrapposizione, nel senso che le immaigni saranno sovrapposte ALMENO questo valore
    matching_method = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    """
    #print("matching method: " + matching_method)
    method = matching_method

    h, w, channels = imgBottom.shape
    _y = 0
    _h = int(h*percent_overlap/100)
    _x = 0
    _w = w

    template = imgBottom[_y:_y+_h, _x:_x+_w]

    res = cv2.matchTemplate(imgTop, template, method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(min_val, max_val, min_loc, max_loc)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    #cv2.rectangle(imgTop, top_left, bottom_right, 255, 2)

    #print(imgTop.shape)
    stitched = cv2.resize(imgTop, (imgTop.shape[1], imgTop.shape[0]+h-(imgTop.shape[0]-top_left[1])))
    #print(stitched.shape)
    stitched[0:imgTop.shape[0], 0:imgTop.shape[1]] = imgTop
    stitched[top_left[1]:top_left[1]+imgBottom.shape[0], 0:imgBottom.shape[1]] = imgBottom

    return stitched

def stitch_fast(imgTop, imgBottom, search_roi_x, search_roi_w, percent_overlap=50.0, matching_method=cv2.TM_SQDIFF_NORMED):
    """
    percent_overlap = minima percentuale di sovrapposizione, nel senso che le immaigni saranno sovrapposte ALMENO questo valore
    matching_method = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    """
    ok, offset_y = find_stitch_offset(imgTop, imgBottom, search_roi_x, search_roi_w, percent_overlap, matching_method)

    return stitch_offset(imgTop, imgBottom, offset_y)

def stitch_offset(imgTop, imgBottom, offset_y):
    h, w, channels = imgBottom.shape
    stitched = np.zeros((imgTop.shape[0]+h-(imgTop.shape[0]-offset_y), imgTop.shape[1], 3), np.uint8) #cv2.resize(imgTop, (imgTop.shape[1], imgTop.shape[0]+h-(imgTop.shape[0]-y)))
    #print(stitched.shape)
    stitched[0:imgTop.shape[0], 0:imgTop.shape[1]] = imgTop
    stitched[offset_y:offset_y+imgBottom.shape[0], 0:imgBottom.shape[1]] = imgBottom
    return stitched

def find_stitch_offset(imgTop, imgBottom, search_roi_x, search_roi_w, percent_overlap=50.0, matching_method=cv2.TM_SQDIFF_NORMED):
    method = matching_method

    h, w, channels = imgBottom.shape
    _y = 0
    _h = int(h*percent_overlap/100)
    _x = 0
    _w = w

    template = imgBottom[_y:_y+_h, search_roi_x:search_roi_w]

    #res = cv2.matchTemplate(imgTop, template, method)

    resize = False
    i = None
    if resize:
        """
        RESIZE permette notevole risparmio di tempo
        """
        template = cv2.resize(template, (1,template.shape[0]))
        i = cv2.resize(imgTop[:,search_roi_x:search_roi_w], (1,imgTop[:,search_roi_x:search_roi_w].shape[0]))
    else:
        i=imgTop[:,search_roi_x:search_roi_w]
    ok, x, y, similarity_result = match_template(template, i, similarity=0.03)

    #ok, x, y = match_template(template, imgTop[:,search_roi_x:search_roi_w], similarity=0.5)
    return ok, y,similarity_result

def match_template(template, image, similarity=0.1, matching_method=cv2.TM_SQDIFF_NORMED):
    #ok, results = match_template_all(template, image, similarity, matching_method)
    #return ok, results[0][1], results[0][0]
    method = matching_method
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ok = False
    res = cv2.matchTemplate(image, template, method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #print("match_template - min_val %s - max_val %s - similarity %s"%(min_val, max_val, similarity))    
    #print(min_val, max_val, min_loc, max_loc)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    similarity_result = 0
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        ok = (min_val < similarity)
        similarity_result = min_val
        #print("min_val:    %s"%min_val)
        #loc = np.where( res <= similarity)
        #results = list(zip(*loc[::-1]))
    else:
        top_left = max_loc
        ok = (max_val > similarity)
        similarity_result = max_val
        #print("max_val:    %s"%max_val)
        #loc = np.where( res >= similarity)
        #results = list(zip(*loc[::-1]))
    x = top_left[0]
    y = top_left[1]

    return ok,x,y,similarity_result

def match_template_all(template, image, similarity=0.1, matching_method=cv2.TM_SQDIFF_NORMED):
    """
    percent_overlap = minima percentuale di sovrapposizione, nel senso che le immaigni saranno sovrapposte ALMENO questo valore
    matching_method = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    """
    #print("matching method: " + matching_method)
    method = matching_method
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #res e' un'immagine in livelli di grigio, in base al metodo di matching scelto, 
    # il punto di buon matching puo' essere piu' chiaro o piu' scuro
    res = cv2.matchTemplate(image, template, method)
    
    # convert the grayscale image to binary image
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    threshold_method = 1 if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else 0
    ret,thresh = cv2.threshold(res,similarity,255,threshold_method)


    # calculate moments of binary image
    #M = cv2.moments(thresh)
    #print("moments")
    #print(M)

    #definiamo parametri per selezione dei blob
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # Change thresholds
    params.minThreshold = 100    # the graylevel of images
    params.maxThreshold = 255

    params.filterByColor = False
    #params.blobColor = 255

    # Filter by Area
    params.filterByArea = False
    params.minArea = 1

    detector = cv2.SimpleBlobDetector_create(params) #SimpleBlobDetector()
    
    # Detect blobs.
    keypoints = detector.detect(thresh.astype(np.uint8))
    print("keypoints>>>>")
    print(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(cv2.cvtColor(thresh.astype(np.uint8), cv2.COLOR_GRAY2BGR), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    results = []
    for k in keypoints:
        results.append(k.pt)
    
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print("match_template - min_val %s - max_val %s - similarity %s"%(min_val, max_val, similarity))    
    #print(min_val, max_val, min_loc, max_loc)

    res2 = np.reshape(res, res.shape[0]*res.shape[1])
    sort = np.argsort(res2)
    #(y1, x1) = np.unravel_index(sort[0], res.shape) # best match
    #(y2, x2) = np.unravel_index(sort[1], res.shape) # second best match  
    results = []
    print(min_val)
    for r in range(1,5):#sort:
        results.append(np.unravel_index(sort[r], res.shape))
        x = results[-1][1]
        y = results[-1][0]
        print(x, y, res[y,x])
    
    #print(results)
    """
    ok = len(results)>0

    return ok,results#,thresh


def perspective_correction(img, w_perspective, h_perspective):
    h = img.shape[0]
    w = img.shape[1]
    #channels = img.shape[2]

    w_top = max(0, w*(-w_perspective))*0.5
    w_bottom = max(0, w*w_perspective)*0.5

    h_left = max(0, h*(-h_perspective))*0.5
    h_right = max(0, h*h_perspective)*0.5

    pts_src = np.array([[0,0], [w, 0], [w, h], [0, h]])

    pts_dst = np.array([[-w_top, -h_left], [w + w_top, -h_right], [w + w_bottom, h + h_right], [-w_bottom, h + h_left]])
    #pts_dst = np.array([[+w_top, +h_left], [w - w_top, -h_right], [w - w_bottom, h - h_right], [+w_bottom, h - h_left]])
    homography, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    #im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    img_result = cv2.warpPerspective(img, homography, (img.shape[1],img.shape[0]))
    return img_result


def get_caltab_points(image_filename_list, output_image_widget, caltab_w=7, caltab_h=6):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((caltab_h*caltab_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:caltab_w,0:caltab_h].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    print("CAMERA CALIBRATION in progress...")
    for fname in image_filename_list:
        print("processing_image %s"%fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (caltab_w,caltab_h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)#,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("    ok")
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (caltab_w,caltab_h), corners2,ret)
            output_image_widget.refresh(img)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
        else:
            discard_folder = "./discarded/"
            move_to = discard_folder + fname.split("/")[-1].split("\\")[-1]
            print("    discarded. moved to:" + move_to)
            try:
                shutil.move(fname, move_to)
            except Exception:
                os.mkdir(discard_folder)
                shutil.move(fname, move_to)
            
            
    return imgpoints, objpoints

def threshold(image):
    #cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    #b, g, r = cv2.split(img)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

def histogram_equalize(image):
    b, g, r = cv2.split(image)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))
    """
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(image)
    #h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)
    """


def get_histograms_rgb(image):
    b = cv2.calcHist([image],[0],None,[256],[0,256]).ravel()
    g = cv2.calcHist([image],[1],None,[256],[0,256]).ravel()
    r = cv2.calcHist([image],[2],None,[256],[0,256]).ravel()
    return [r, g, b]

def get_histograms_hsv(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).ravel()
    return [h, s, v]

def canny(image, min_val=100, max_val=200):
    return cv2.Canny(image, min_val, max_val)


#FILTER HISTOGRAM COLOR HSV
"""
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#Compute histogram
hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
#Convert histogram to simple list
hist = [val[0] for val in hist]
#Generate a list of indices
indices = list(range(0, 180))
#Descending sort-by-key with histogram value as key
#s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
s = [x for y,x in sorted(zip(hist,indices), reverse=True)]
if 0 in s:
    s.remove(0)
if 255 in s:
    s.remove(255)

#Index of highest peak in histogram
index_of_highest_peak = s[0]
print(index_of_highest_peak)
#Index of second highest peak in histogram
index_of_second_highest_peak = s[1]
print(index_of_second_highest_peak)

hlow = index_of_highest_peak - 0
hlow = hlow if hlow > 0 else (180+hlow)
hhigh = (index_of_highest_peak + 0)%180
color_light = (hlow, 0, 0)
color_dark = (hhigh, 255, 255)
mask = cv2.inRange(hsv, color_light, color_dark)
hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
#hsv = cv2.bitwise_or(hsv, hsv, mask=cv2.bitwise_not(mask))
img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
app.add_miniature(img)

img_thresh = image_utils.threshold(img)
app.add_miniature(img_thresh)
#img_thresh = cv2.bitwise_not(img_thresh)

#img = image_utils.histogram_equalize(img)
#app.add_miniature(img)
"""

""" Trovare massimi in istogramma per poi filtrare il resto con cv2.inRange
#Compute histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

#Convert histogram to simple list
hist = [val[0] for val in hist]; 

#Generate a list of indices
indices = list(range(0, 256));

#Descending sort-by-key with histogram value as key
s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]

#Index of highest peak in histogram
index_of_highest_peak = s[0][0];

#Index of second highest peak in histogram
index_of_second_highest_peak = s[1][0];

"""
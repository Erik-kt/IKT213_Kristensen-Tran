import cv2
import numpy as np
import matplotlib.pyplot as plt

def HCD(reference_image):

    img = cv2.imread(reference_image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    threshold = 0.05 * dst.max()
    mask = dst > threshold

    dst_nms = cv2.dilate(dst,None)
    mask &= (dst == dst_nms)

    corners = np.argwhere(mask)

    for y, x in corners:
        img[y,x] = [0,0,255]  # radius=2 pixels

    #for preview
    #cv2.imshow('dst',img)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
    cv2.imwrite('harris.png',img)
    print('Saved img as harris.png')




def align_image(image_to_align, reference_image, max_features=10, good_match_precent=0.7):

    MIN_MATCH_COUNT = max_features

    img1 = cv2.imread(image_to_align, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None) # align
    kp2, des2 = sift.detectAndCompute(img2, None) # ref

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)


    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2_draw = img2.copy()
        img2_draw = cv2.polylines(img2_draw,[np.int32(dst)],True, 255, 3,cv2.LINE_AA)

        aligned = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        cv2.imwrite('aligned.png', aligned)
        print('Saved aligned image as aligned.png')
    else:
        print('No good matches - {}/{}'.format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        img2_draw = img2

    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    #plt.imshow(img3, 'gray'),plt.show() # Displays matches

    cv2.imwrite('matches.png', img3)
    print('Saved matches image as matches.png')


if __name__ == "__main__":
    HCD('reference_img.png')
    align_image('align_this.jpg', 'reference_img.png')
import os
import cv2
sifttrans = []
for file in os.listdir("SOCOFing\\Real")[:6500]:
    aaa = cv2.imread("SOCOFing\\Real\\"+ file)
    filee=file[:-4]
    a = cv2.cvtColor(aaa, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypnt, destor = sift.detectAndCompute(a, None)
    sifttrans.append([keypnt, destor, a,filee ])
    
def kiemtra(keypoints_1, descriptors_1, kp_reall, best_scoree, filenamee, imagee_real, mpp, mlem, mlemm):
    #filenamet = None
    #imaget = None
    #keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    for file in sifttrans:
    #for file in os.listdir("SOCOFing\\Real")[:6500]:
        #if counter%10 == 0:
        #    print(counter)
        #print(file)
        #print(file)
        #if file == "1__M_Left_index_finger.BMP":
        #   print(file)
        #  break
        #    counter += 1
        #fingerprint_image = cv2.imread("SOCOFing\\Real\\"+ file)
        fingerprint_image = file[2]
        #print(fingerprint_image)
        #sift = cv2.SIFT_create()

        
        #keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher({'algorithm':1, 'trees':10},
                                        {}).knnMatch(descriptors_1, file[1], k=2)
        
        match_points = []

        for p, q in matches:
            #print(p.distance, '', q.distance)
            if p.distance <0.7 * q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) < len(file[0]):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(file[0])
        if len(match_points) / keypoints*10 >= best_scoree:
            #print(best_score)
            best_scoree = len(match_points)/keypoints*10
            filenamee = file[3]
            imagee_real = fingerprint_image
            mlem = len(match_points)
            mlemm = keypoints
            kp_reall, mpp = file[0], match_points
        #sodem+=1
    return kp_reall, best_scoree, filenamee, imagee_real, mpp, mlem, mlemm


#ten_anh=""
def xacthuc(tenanh):
    best_score = 0
    filename = None
    image_real = None
    kp_real, mp = None, None
    lem=0
    lemm=0
    #bb=cv2.imread("SOCOFing\\Altered\\Altered-Medium\\"+tenanh)
    #print(tenanh)
    bb=cv2.imread(tenanh)
    b = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypont, destoor = sift.detectAndCompute(b, None)
    kp_real, best_score, filename, image_real, mp, lem, lemm=kiemtra( keypont, destoor,kp_real , best_score, filename, image_real, mp,lem, lemm)
    print(filename, " ", tenanh," ", lem, " ", lemm," ", best_score)
    if(filename in tenanh):
        #cayvaidai=cv2.imread()
        result = cv2.drawMatches(b, keypont, image_real, kp_real, mp, None)
        result = cv2.resize(result, None, fx=4, fy=4)
        cv2.imshow("results", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

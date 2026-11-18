# =====================================================================
# Computer Vision Lab (ADPEC705) - Experiments 1 to 10
# Every experiment is fully independent: it has its own imports and
# loads its own image. Copy any one block into a new file and it runs.
#
# Install once:  pip install opencv-python numpy
# Image:         put your photo next to this file as "image.jpg"
#                (if it is missing, a sample image is generated automatically)
# Press any key to close each image window.
# =====================================================================


# ---------------------------------------------------------------------
# 1. Perform basic image handling and processing operations on the image.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    cv2.circle(img, (350, 300), 30, (0, 0, 200), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)  # shrink big photos so they fit the screen

print("Shape:", img.shape, "| Data type:", img.dtype, "| Size:", img.size)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
resized = cv2.resize(img, None, fx=0.5, fy=0.5)
h, w = img.shape[:2]
cropped = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
flipped = cv2.flip(img, 1)
blurred = cv2.GaussianBlur(img, (7, 7), 0)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
equalized = cv2.equalizeHist(gray)
negative = 255 - img

cv2.imwrite("exp1_gray.jpg", gray)

for name, im in [("Original", img), ("Grayscale", gray), ("HSV", hsv), ("Resized", resized),
                 ("Cropped", cropped), ("Flipped", flipped), ("Blurred", blurred),
                 ("Threshold", thresh), ("Equalized", equalized), ("Negative", negative)]:
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 2. Perform geometric transformations (translation, rotation, scaling, shear, affine).
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    cv2.circle(img, (350, 300), 30, (0, 0, 200), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

h, w = img.shape[:2]

M_translate = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M_translate, (w, h))

M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
rotated = cv2.warpAffine(img, M_rotate, (w, h))

scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

M_shear = np.float32([[1, 0.3, 0], [0, 1, 0]])
sheared = cv2.warpAffine(img, M_shear, (int(w * 1.3), h))

src = np.float32([[50, 50], [200, 50], [50, 200]])
dst = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(src, dst)
affine = cv2.warpAffine(img, M_affine, (w, h))

print("Translation matrix:\n", M_translate)
print("Rotation matrix:\n", M_rotate)
print("Affine matrix:\n", M_affine)

for name, im in [("Original", img), ("Translated", translated), ("Rotated", rotated),
                 ("Scaled", scaled), ("Sheared", sheared), ("Affine", affine)]:
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 3. Compute the homography matrix between two sets of corresponding points.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    cv2.circle(img, (350, 300), 30, (0, 0, 200), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

h, w = img.shape[:2]

# corresponding points: source (a tilted quadrilateral) -> destination (full rectangle)
# points are given as fractions of the image size, so they work for any image
src_pts = np.float32([[0.21 * w, 0.16 * h], [0.80 * w, 0.24 * h],
                      [0.86 * w, 0.84 * h], [0.16 * w, 0.76 * h]])
dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

H, mask = cv2.findHomography(src_pts, dst_pts)
print("Homography matrix (3x3):\n", H)

warped = cv2.warpPerspective(img, H, (w, h))
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Warped using Homography", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 4. Apply perspective transformation to get a top-down view of an image.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    cv2.circle(img, (350, 300), 30, (0, 0, 200), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

h, w = img.shape[:2]

# four corners (top-left, top-right, bottom-right, bottom-left) as fractions of image size
src = np.float32([[0.21 * w, 0.16 * h], [0.80 * w, 0.24 * h],
                  [0.86 * w, 0.84 * h], [0.16 * w, 0.76 * h]])
width, height = 600, 400
dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

M = cv2.getPerspectiveTransform(src, dst)
print("Perspective matrix:\n", M)
result = cv2.warpPerspective(img, M, (width, height))

marked = img.copy()
for p in src:
    cv2.circle(marked, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)

cv2.imshow("Original (corners marked)", marked)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Perspective Transformed", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 5. Perform camera calibration using chessboard images.
# (uses photos from folder "calib_images" if present, otherwise generates test chessboard images)
# ---------------------------------------------------------------------
import glob
import cv2
import numpy as np

CHESSBOARD = (9, 6)  # number of INNER corners (columns, rows)

# --- collect images ---
files = glob.glob("calib_images/*.jpg") + glob.glob("calib_images/*.png")
images = [cv2.imread(f) for f in files]
images = [im for im in images if im is not None]

if not images:
    print("No images in 'calib_images/' -> generating synthetic chessboard images.")
    sq = 50
    board = np.full(((CHESSBOARD[1] + 3) * sq, (CHESSBOARD[0] + 3) * sq), 255, np.uint8)
    for r in range(CHESSBOARD[1] + 1):
        for c in range(CHESSBOARD[0] + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    bh, bw = board.shape
    base = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    shifts = [[(0, 0), (0, 0), (0, 0), (0, 0)],
              [(60, 40), (-40, 0), (-20, -40), (20, 20)],
              [(-40, 20), (40, 70), (40, -40), (-20, -20)],
              [(40, 0), (0, 90), (-70, -20), (20, -70)],
              [(0, 70), (-70, 20), (0, -70), (70, -20)],
              [(90, 20), (-20, -20), (-40, -70), (0, 40)],
              [(-70, -40), (20, 40), (70, -20), (-20, 70)],
              [(30, 80), (-80, 30), (-30, -80), (80, -30)]]
    for sh in shifts:
        dst = base + np.float32(sh)
        Mw = cv2.getPerspectiveTransform(base, dst)
        warped = cv2.warpPerspective(board, Mw, (bw, bh), borderValue=255)
        images.append(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))

# --- find chessboard corners in every image ---
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

objpoints, imgpoints, size = [], [], None
for im in images:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        shown = im.copy()
        cv2.drawChessboardCorners(shown, CHESSBOARD, corners, found)
        cv2.imshow("Detected corners", shown)
        cv2.waitKey(300)
cv2.destroyAllWindows()
print("Chessboard found in", len(objpoints), "of", len(images), "images")

# --- calibrate ---
if len(objpoints) >= 3:
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None, flags=flags)
    print("RMS reprojection error:", ret)
    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    im = images[0]
    undistorted = cv2.undistort(im, K, dist)
    cv2.imshow("Original", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough chessboard detections. Check the CHESSBOARD size (inner corners).")


# ---------------------------------------------------------------------
# 6. Compute the fundamental matrix from two views of the same scene.
# (uses view1.jpg and view2.jpg if present, otherwise generates two views)
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img1 = cv2.imread("view1.jpg") if os.path.exists("view1.jpg") else None
img2 = cv2.imread("view2.jpg") if os.path.exists("view2.jpg") else None

if img1 is None or img2 is None:
    print("view1.jpg / view2.jpg not found -> generating two views from a sample image.")
    img1 = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
    if img1 is None:
        rng = np.random.RandomState(0)
        img1 = np.full((500, 700, 3), 60, np.uint8)
        cv2.fillPoly(img1, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
        for _ in range(60):  # random shapes give plenty of features
            x, y = rng.randint(120, 580), rng.randint(100, 400)
            col = tuple(int(v) for v in rng.randint(0, 255, 3))
            if rng.rand() > 0.5:
                cv2.circle(img1, (x, y), rng.randint(5, 20), col, -1)
            else:
                cv2.rectangle(img1, (x, y), (x + rng.randint(8, 30), y + rng.randint(8, 30)), col, -1)
    s = 800 / max(img1.shape[:2])
    if s < 1:
        img1 = cv2.resize(img1, None, fx=s, fy=s)
    h, w = img1.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[20, 15], [w - 10, 0], [w - 25, h - 10], [5, h - 30]])
    img2 = cv2.warpPerspective(img1, cv2.getPerspectiveTransform(src, dst), (w, h))

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(g1, None)
k2, d2 = sift.detectAndCompute(g2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(d1, d2, k=2)

pts1, pts2 = [], []
for pair in matches:
    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
        pts1.append(k1[pair[0].queryIdx].pt)
        pts2.append(k2[pair[0].trainIdx].pt)
pts1, pts2 = np.float32(pts1), np.float32(pts2)
print("Good matches:", len(pts1))

if len(pts1) >= 8:
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    print("Fundamental matrix F:\n", F)
    print("Rank of F (should be 2):", np.linalg.matrix_rank(F))

    inl = mask.ravel() == 1
    pts1, pts2 = pts1[inl], pts2[inl]
    print("Inlier points:", len(pts1))

    # draw epipolar lines in image 1 for the points of image 2
    lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    out = img1.copy()
    c = out.shape[1]
    for r, p in list(zip(lines, pts1))[:20]:
        color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
        if abs(r[1]) > 1e-9:
            y0 = int(-r[2] / r[1])
            y1 = int(-(r[2] + r[0] * c) / r[1])
            cv2.line(out, (0, y0), (c, y1), color, 1)
        cv2.circle(out, (int(p[0]), int(p[1])), 5, color, -1)
    cv2.imshow("Epipolar lines (Image 1)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Need at least 8 matches to compute F. Use two clearer photos of the same scene.")


# ---------------------------------------------------------------------
# 7. Perform edge detection, line detection and corner detection.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    for i in range(6):
        cv2.line(img, (200, 160 + i * 38), (520, 175 + i * 38), (30, 30, 30), 3)
    cv2.rectangle(img, (250, 220), (320, 270), (200, 100, 0), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Edge detection ---
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
laplacian = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
canny = cv2.Canny(blur, 50, 150)

# --- Line detection (Hough transform) ---
lines_img = img.copy()
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
count = 0
if lines is not None:
    for l in np.array(lines).reshape(-1, 4):
        cv2.line(lines_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 2)
        count += 1
print("Lines detected:", count)

# --- Corner detection ---
harris_img = img.copy()
harris = cv2.cornerHarris(np.float32(blur), 2, 3, 0.04)
harris = cv2.dilate(harris, None)
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]

shi_img = img.copy()
corners = cv2.goodFeaturesToTrack(blur, 100, 0.01, 10)
if corners is not None:
    for c in np.array(corners).reshape(-1, 2):
        cv2.circle(shi_img, (int(c[0]), int(c[1])), 4, (0, 255, 0), -1)
    print("Shi-Tomasi corners:", len(np.array(corners).reshape(-1, 2)))

for name, im in [("Original", img), ("Sobel", sobel), ("Laplacian", laplacian),
                 ("Canny", canny), ("Hough Lines", lines_img),
                 ("Harris Corners", harris_img), ("Shi-Tomasi Corners", shi_img)]:
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 8. Extract SIFT feature descriptors and match them between two images.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    rng = np.random.RandomState(0)
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    for _ in range(60):  # random shapes give plenty of features
        x, y = rng.randint(120, 580), rng.randint(100, 400)
        col = tuple(int(v) for v in rng.randint(0, 255, 3))
        if rng.rand() > 0.5:
            cv2.circle(img, (x, y), rng.randint(5, 20), col, -1)
        else:
            cv2.rectangle(img, (x, y), (x + rng.randint(8, 30), y + rng.randint(8, 30)), col, -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
print("Keypoints found:", len(kp))
print("Descriptor shape:", None if des is None else des.shape, "(each descriptor is 128-D)")

out = cv2.drawKeypoints(img, kp, None, flags=4)  # 4 = draw rich keypoints (size + orientation)
cv2.imshow("SIFT Keypoints", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# match with a rotated copy of the same image
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 0.9)
img2 = cv2.warpAffine(img, M, (w, h))
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des, des2, k=2)
good = [p[0] for p in matches if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
print("Good matches (ratio test):", len(good))

result = cv2.drawMatches(img, kp, img2, kp2, good[:50], None, flags=2)
cv2.imshow("SIFT Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 9. Extract SURF and HOG feature descriptors from the image.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    for i in range(6):
        cv2.line(img, (200, 160 + i * 38), (520, 175 + i * 38), (30, 30, 30), 3)
    cv2.circle(img, (350, 300), 30, (0, 0, 200), -1)
s = 800 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- SURF (patented: missing in most OpenCV installs, so ORB is used automatically instead) ---
try:
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(gray, None)
    label = "SURF"
except Exception:
    print("SURF is not available in this OpenCV build -> using ORB instead.")
    kp, des = cv2.ORB_create(500).detectAndCompute(gray, None)
    label = "ORB (SURF substitute)"
print(label, "keypoints:", len(kp))
cv2.imshow(label + " Keypoints", cv2.drawKeypoints(img, kp, None, (0, 255, 0), 4))
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- HOG feature vector (OpenCV) ---
resized = cv2.resize(gray, (128, 128))
hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
features = hog.compute(resized)
print("HOG feature vector length:", features.size)

# --- HOG visualisation (drawn manually: 8x8 cells, 9 orientation bins) ---
gx = cv2.Sobel(resized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(resized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
ang = ang % 180
vis = np.zeros((128, 128), np.uint8)
for cy in range(0, 128, 8):
    for cx in range(0, 128, 8):
        m = mag[cy:cy + 8, cx:cx + 8].ravel()
        a = ang[cy:cy + 8, cx:cx + 8].ravel()
        hist, _ = np.histogram(a, bins=9, range=(0, 180), weights=m)
        hist = hist / (hist.max() + 1e-6)
        for k, v in enumerate(hist):
            theta = np.deg2rad(k * 20 + 10 + 90)  # edge direction is perpendicular to gradient
            dx, dy = np.cos(theta) * 4 * v, np.sin(theta) * 4 * v
            p1 = (int(cx + 4 - dx), int(cy + 4 - dy))
            p2 = (int(cx + 4 + dx), int(cy + 4 + dy))
            cv2.line(vis, p1, p2, 255, 1)
vis = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_NEAREST)
cv2.imshow("HOG Visualisation", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# 10. Project: Smart Document Scanner using edge detection, contours and perspective transform.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np

img = cv2.imread("image.jpg") if os.path.exists("image.jpg") else None
if img is None:
    img = np.full((500, 700, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 80], [560, 120], [600, 420], [110, 380]])], (240, 240, 240))
    for i in range(6):
        cv2.line(img, (200, 160 + i * 38), (520, 175 + i * 38), (30, 30, 30), 3)

orig = img.copy()
ratio = 500.0 / img.shape[0]
small = cv2.resize(img, None, fx=ratio, fy=ratio)

# Step 1: pre-processing and edge detection
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 50, 150)
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# Step 2: find the largest 4-corner contour (the document)
res = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = res[0] if len(res) == 2 else res[1]  # works for both OpenCV 3 and 4+
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
doc = None
for c in cnts:
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        doc = approx
        break

if doc is None:
    print("Document outline not found. Use a photo where the whole paper is visible on a darker background.")
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    outline = small.copy()
    cv2.drawContours(outline, [doc], -1, (0, 255, 0), 3)

    # Step 3: order the 4 corners (TL, TR, BR, BL) and apply perspective transform
    pts = (doc.reshape(4, 2) / ratio).astype("float32")
    sm = pts.sum(axis=1)
    df = np.diff(pts, axis=1).ravel()
    rect = np.array([pts[np.argmin(sm)], pts[np.argmin(df)],
                     pts[np.argmax(sm)], pts[np.argmax(df)]], dtype="float32")
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (W, H))

    # Step 4: scanned black-and-white look
    wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(wg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)
    cv2.imwrite("scanned_output.jpg", scanned)
    print("Saved scanned_output.jpg")

    for name, im in [("1. Original", small), ("2. Edges", edges), ("3. Outline", outline),
                     ("4. Warped", warped), ("5. Final Scan", scanned)]:
        cv2.imshow(name, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

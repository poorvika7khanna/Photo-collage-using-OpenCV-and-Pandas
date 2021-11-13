import cv2
import pandas as pd

img1 = cv2.imread("bottom_left.jpg")
if img1 is not None:
    imbl = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    bottom_left = cv2.resize(imbl, (200, 200))
img2 = cv2.imread("bottom_right.jpg")
imbr = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
bottom_right = cv2.resize(imbr, (200, 200))
bottom_right = cv2.copyMakeBorder(bottom_right, 10, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
img3 = cv2.imread("center.jpeg")
imc = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
center = cv2.resize(imc, (100, 100))
center = cv2.copyMakeBorder(center, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
img4 = cv2.imread("top_left.jpg")
imtl = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
top_left = cv2.resize(imtl, (200, 200))
top_left = cv2.copyMakeBorder(top_left, 0, 10, 0, 0, cv2.BORDER_CONSTANT, value=0)
img5 = cv2.imread("top_right.jpg")
imtr = cv2.cvtColor(img5,cv2.COLOR_BGR2RGB)
top_right = cv2.resize(imtr, (200, 200))
sample1 = cv2.vconcat([top_left, bottom_left])
sample1 = cv2.copyMakeBorder(sample1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
sample2 = cv2.vconcat([top_right, bottom_right])
sample2 = cv2.copyMakeBorder(sample2, 10, 10, 0, 10, cv2.BORDER_CONSTANT, value=0)
sample = cv2.hconcat([sample1, sample2])
final = sample.copy()
final[155:275, 155:275, :] = center
final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

cv2.imwrite("collage.jpg", final)

r = final[:,:,0].reshape((-1,))
g = final[:,:,1].reshape((-1,))
b = final[:,:,2].reshape((-1,))

df = pd.DataFrame({
    'r': r,
    'g': g,
    'b': b
})

df.to_csv("final.csv", index=False)

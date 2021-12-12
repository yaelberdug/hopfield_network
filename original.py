# random stored pattern
# Add noise to first stored pattern and let Hopfield to converge to steady state
# Measure average error.

# https://github.com/ctawong/hopfield
# cmd line: -t 1.PNG 2.PNG 3.PNG 4.PNG 5.PNG 6.PNG  -p pre.png -s 32 --noise-prob 0.06 --n-repeats 1000 --max-n-store 10

from models import Hopfield, OriginalHopfield, ModernHopfield, addNoise
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import cv2


def convert_image_to_32(path, image_name):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dim = (32, 32)
    # resize image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    thresh = 127
    im_bw = cv2.threshold(resized_img, thresh, 255, cv2.THRESH_BINARY)[1]
    plt.plot()
    plt.imshow(im_bw, cmap='binary')
    cv2.imwrite(path.replace('.jpg', '_32b.png'), im_bw)


def getOptions():
    parser = argparse.ArgumentParser(description='Parses Command.')
    parser.add_argument('-t', '--train', nargs='*', help='Training data directories.')
    parser.add_argument('-p', '--predict', nargs='*', help='Predict image.')
    parser.add_argument('-s', '--size', type=int, help='Image size nXn.')
    parser.add_argument('--n-repeats', type=int, default=10, help='Number of repeats for each number of memory runs')
    parser.add_argument('--noise-prob', type=float, default=0.1, help='Noise probability for corrupting the query pattern')
    parser.add_argument('--max-n-store', type=int, default=20, help='Max number of stored patterns')
    options = parser.parse_args(sys.argv[1:])
    return options


def compute(h_model, x_mat):
    h_model.set(x_mat)
    n = h_model.update()
    diff = np.abs((x_mat - h_model.neurons)).sum() / 2   # np.abs((x_mat - h_model.neurons)).sum(1) / 2
    min_diff.append(np.min(diff))
    images_store.append(np.reshape(h_model.neurons, (img.shape[0], img.shape[1])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Model initialized with weights shape ')


# parser = argparse.ArgumentParser()
# parser.add_argument('--N', type =int, default=100, help='Number of neurons in Hopfield net')
# parser.add_argument('--n-repeats', type =int, default=10, help='Number of repeats for each number of memory runs')
# parser.add_argument('--noise-prob', type =float, default=0.1, help='Noise probability for corrupting the query pattern')
# parser.add_argument('--max-n-store', type =int, default=20, help='Max number of stored patterns')

opt = getOptions()  #parser.parse_args()
print(opt)
new_images = ['1_new_32b', '2_new_32b', '1_32b']   # ['1_32b', '2_32b']

N = opt.size  # opt.N
n_repeats = opt.n_repeats
noise_prob = opt.noise_prob
max_n_store = opt.max_n_store

# min_diff = {}
global images_store
images_store = []
global min_diff
min_diff = []
query_store = []
X_store = []

for r in opt.train:   #(n_repeats):
    #X_store = np.random.choice([-1, 1], size=[n_store, N])
    print('Start training ', r, '...')
    img = plt.imread(r)
    img = np.mean(img, axis=2)
    if img.shape != (opt.size, opt.size):
        print('Error: image shape ', img.shape, ' is not (', opt.size, ',', opt.size, ')')
    img_mean = np.mean(img)
    img = np.where(img < img_mean, -1, 1)
    reshaped_img = np.reshape(img, (1, img.shape[0]*img.shape[1]))
    X_store.append(reshaped_img[0])
    x_query = reshaped_img[0].copy()
    x_query = addNoise(x_query, prob=noise_prob)
    query_store.append(x_query)


    # learn network for each pattern and compute:
    # compute original hopfield:
    h_orig = OriginalHopfield(reshaped_img)
    # compute modern hopfield:
    h_modern = ModernHopfield(reshaped_img)
    compute(h_orig, x_query)
    compute(h_modern, x_query)
    # ----


# learn all patterns before trying to retrieve corrupted patterns :
X_store_arr = np.array(X_store)
# compute original hopfield:
h_orig = OriginalHopfield(X_store_arr)
# compute modern hopfield:
h_modern = ModernHopfield(X_store_arr)

for iter in query_store:
    compute(h_orig, iter)
    compute(h_modern, iter)

# Add new images and try to identify them using last hopfield network:
for iter in new_images:
    img = plt.imread(fr'New_Figures\{iter}.png')    # .jpg')
    print('Start training ', iter, '...')
    # img = np.mean(img, axis=2)
    if img.shape != (opt.size, opt.size):
        print('Error: image shape ', img.shape, ' is not (', opt.size, ',', opt.size, ')')
    img_mean = np.mean(img)
    img = np.where(img < 1, -1, 1)
    x_query = np.reshape(img, (1, img.shape[0]*img.shape[1]))[0]   # np.reshape(img, (1, img.shape[0] * img.shape[1]))[0]
    compute(h_orig, x_query)
    compute(h_modern, x_query)


print('n_memories, average_error_rate')
for n_store in min_diff:  # range(1, max_n_store+1):
    print('%d, %1.3f' % (n_store, np.mean(n_store) / N))
for my_index in range(int(len(images_store) / 2)):
    img_1, img_2 = images_store[my_index * 2 + 1], images_store[my_index * 2 + 1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].set_title('Original Hopfield Output')
    ax[0].imshow(img_1 * 255, cmap='binary')
    ax[1].set_title('Modern Hopfield Output')
    ax[1].imshow(img_2 * 255, cmap='binary')
    plt.show()

print('stop')


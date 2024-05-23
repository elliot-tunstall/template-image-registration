import matplotlib.pyplot as plt

def show_image(img, pixelMap):
    plt.figure()
    plt.imshow(img, extent=(pixelMap['X'].min(), pixelMap['X'].max(), pixelMap['Z'].max(), pixelMap['Z'].min()), origin='upper')
    plt.colorbar()


def show_image_pts(img, pixelMap, pts, index=2):
    plt.figure()
    plt.imshow(img, extent=(pixelMap['X'].min(), pixelMap['X'].max(), pixelMap['Z'].max(), pixelMap['Z'].min()), origin='upper')
    plt.scatter(pts[:,0], pts[:,index],s=2,c='k',zorder=2)
    plt.colorbar()

def compare_images(set, pixelMap):
    plt.figure()
    for i in range(len(set)):
        plt.subplot(1,len(set),i+1)
        plt.imshow(set[i], extent=(pixelMap['X'].min(), pixelMap['X'].max(), pixelMap['Z'].max(), pixelMap['Z'].min()), origin='upper')
        plt.colorbar()
    plt.show(block=False)
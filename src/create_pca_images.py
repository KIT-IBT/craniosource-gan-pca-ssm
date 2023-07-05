import argparse
import os.path
import pathlib

import numpy
import numpy.random
from sklearn import decomposition
import skimage.io

def parseargs():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default=os.path.expanduser("../dataset28/validation"))
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory", default=os.path.expanduser("../demo_out/pca"))
    parser.add_argument('-n', '--number', type=int, help="Number images", default=10)
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parseargs()
    classes = ("control", "coronal", "metopic", "sagittal")
    RATIO_VARIANCE=0.95
    pca_estimator = decomposition.PCA(RATIO_VARIANCE)
    rng = numpy.random.default_rng()
    # Create images
    for cl in classes:
        cl_dir = args.output_dir + "/" + cl
        print(cl_dir)
        if not os.path.exists(cl_dir):
            pathlib.Path(cl_dir).mkdir(parents=True)
        class_input_dir = pathlib.Path(os.path.join(args.input_dir,cl))
        img_paths = list(class_input_dir.glob('*.png'))
        imgs = []
        for img_path in sorted([str(p) for p in img_paths]):
            loaded_img = numpy.array(skimage.io.imread(img_path)/255.0)
            imgs.append(loaded_img)
        img_mu = numpy.array(imgs).mean(axis=0)
        imgs = imgs - img_mu
        imgs_vector = imgs.reshape(len(imgs),-1)
        pca_estimator.fit(imgs_vector)
        for i in range(args.number):
            ran = rng.normal(size=len(pca_estimator.components_))
            rev = pca_estimator.inverse_transform(ran/pca_estimator.singular_values_[0] * pca_estimator.explained_variance_ratio_[0])
            img_ran = rev.reshape(loaded_img.shape) + img_mu
            img_uint8 = numpy.uint8(img_ran * 255)
            skimage.io.imsave(cl_dir + f"/drawn_{i}.png",img_uint8)

if __name__ == "__main__":
    main()

import argparse
import os.path
import pathlib
import pickle

import vtk
import numpy
import numpy.random
import skimage.io

import distmaps
import painting

def parseargs():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default=os.path.expanduser("../ssm"))
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory", default=os.path.expanduser("../demo_out/ssm"))
    parser.add_argument('-n', '--number', type=int, help="Number images", default=10)
    args = parser.parse_args()
    return args

def columnvec2xyz(observations):
    axis = 0
    cur_shape = numpy.array(observations).shape
    assert cur_shape[-1] % 3 == 0, f"last axis {axis} not divisible by 3"
    if len(cur_shape) == 1:
        return numpy.reshape(observations,(cur_shape[0]//3,3))
    l = list(cur_shape)
    l[-1] //= 3
    l += [3]
    return numpy.reshape(observations,tuple(l))

def polydata_from_points_cells(points,cells):
    """
    Converts points and cells to vtk.vtkPolydata.
    ----------
    Parameters
    points : [Px3 numpy array] 
    cells : [Cx3 numpy array]
    ----------
    Returns
    polydata : vtk.vtkPolydata
    """
    # This function follows mostly this example:
    # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/Triangle/
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt[0],pt[1],pt[2])
    vtk_cells = vtk.vtkCellArray()
    for cl in cells:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, cl[0])
        triangle.GetPointIds().SetId(1, cl[1])
        triangle.GetPointIds().SetId(2, cl[2])
        vtk_cells.InsertNextCell(triangle)
    # Set polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    return polydata

def main():
    args = parseargs()
    with open(f"{args.input_dir}/template_cells.pickle", "rb") as handle:
        cls_t = pickle.load(handle)
    classes = ("control", "coronal", "metopic", "sagittal")
    painter = painting.ImageFromDistancesCreator()
    for cl in classes:
        with open(f"{args.input_dir}/{cl}.pickle", "rb") as handle:
            sm_dict = pickle.load(handle)
        cur_dir = args.output_dir + f"/{cl}"
        if not os.path.exists(cur_dir):
            pathlib.Path(cur_dir).mkdir(parents=True)
        for i in range(args.number):
            n_components = len(sm_dict['eigenvalues'])
            alpha = numpy.random.normal(size=n_components)
            stat = (sm_dict['eigenvalues'] ** 0.5 * alpha) @ sm_dict['eigenvectors']
            shp = columnvec2xyz(sm_dict['mu'] + stat)
            polydata = polydata_from_points_cells(shp,cls_t)
            lms_inds = distmaps.define_shape_model_landmarks()
            lms_10 = shp[lms_inds]
            cdl = lms_10[distmaps.define_center_landmark_ids(),:]
            intersector = distmaps.MeshIntersector(polydata,cdl)
            creator = distmaps.RayCreator(method="halfsphere",intersector=intersector)
            distances = creator()
            img = painter([distances])[0]
            img_uint8 = numpy.uint8(img * 255)
            skimage.io.imsave(cur_dir + f"/drawn_{i}.png",img_uint8)
            print(f"Done {cl} {i}")

if __name__ == "__main__":
    main()

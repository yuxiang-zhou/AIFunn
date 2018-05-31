import menpo.io as mio
import numpy as np
import menpo.io as mio
import menpo3d.io as m3io
import scipy.io as sio
import sys, traceback

from menpo.feature import no_op, fast_dsift
from menpo.landmark import face_ibug_68_to_face_ibug_49, face_ibug_68_to_face_ibug_68
from menpo3d.camera import PerspectiveCamera
from menpo.transform import AlignmentAffine
# from menpo3d.result import _affine_2d_to_3d

from menpo3d.rasterize import rasterize_mesh
from menpo.transform import rotate_ccw_about_centre
from menpo3d.morphablemodel import ColouredMorphableModel
from menpo.model import PCAModel
from menpo.shape import PointDirectedGraph, PointCloud, ColouredTriMesh, TriMesh
from pathlib import Path
from menpo.visualize import print_progress
from menpo3d.unwrap import optimal_cylindrical_unwrap
from menpo.transform import Translation
from menpo3d.rasterize import rasterize_barycentric_coordinate_images
from menpo.transform import image_coords_to_tcoords
from menpo.shape import TexturedTriMesh

from menpofit.aam import load_balanced_frontal_face_fitter
from menpo3d.morphablemodel.fitter import LucasKanadeMMFitter
from menpo3d.camera import PerspectiveCamera

sys.path.append('/homes/yz4009/wd/gitdev/face2d3d')
from itw3dmm.data import load_tassos_lsfm_combined_model
from itw3dmm import mappings
from shading import lambertian_shading

sys.path.append('/homes/yz4009/wd/gitdev/DeepLearning/Applications/FaceHGNet/detection')
import detect_face
import tensorflow as tf
from scipy import misc

config = tf.ConfigProto(
    device_count = {'GPU': 0}
)

with tf.Graph().as_default() as detect_g:

    sess = tf.Session(graph=detect_g, config=config)
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = detect_face.PNet({'data':data})
            pnet.load('/homes/yz4009/wd/gitdev/DeepLearning/Applications/FaceHGNet/detection/cas1.npy', sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = detect_face.RNet({'data':data})
            rnet.load('/homes/yz4009/wd/gitdev/DeepLearning/Applications/FaceHGNet/detection/cas2.npy', sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = detect_face.ONet({'data':data})
            onet.load('/homes/yz4009/wd/gitdev/DeepLearning/Applications/FaceHGNet/detection/cas3.npy', sess)

        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

minsize = 120 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def detect(img):

    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    bounding_boxes = [PointCloud(b[:4].reshape(-1,2)[:,::-1]).bounding_box() for b in bounding_boxes]

    return bounding_boxes

aam_fitter = load_balanced_frontal_face_fitter()


__model_path = Path('/vol/atlas/homes/aroussos/results/fit3Dto2D/model/ver2016-12-12_LSFMfrmt_maxNpcInf/all_all_all.mat')
shape_model_dict = load_tassos_lsfm_combined_model(__model_path)
shape_model = shape_model_dict['shape_model']
landmarks = m3io.import_landmark_file('../ibug68.ljson').lms.from_vector(shape_model.mean().lms.points[mappings.fw_index_for_lms()])
texture_model = mappings.load_itwmm_texture_fast_dsift_fw()
diagonal = 185
mm = ColouredMorphableModel(shape_model, texture_model, (landmarks),
                            holistic_features=fast_dsift, diagonal=diagonal)
fitter = LucasKanadeMMFitter(mm, n_shape=200, n_texture=200,
                             n_samples=8000, n_scales=1)




def fit(imagepath):

    image = mio.import_image(imagepath, normalize=False)

    if len(image.pixels.shape) == 2:
        image.pixels = np.stack([image.pixels,image.pixels,image.pixels])

    if image.pixels.shape[0] == 1:
        image.pixels = np.concatenate([image.pixels,image.pixels,image.pixels], axis=0)

    print(image.pixels_with_channels_at_back().shape)

    bb = detect(image.pixels_with_channels_at_back())[0]
    initial_shape = aam_fitter.fit_from_bb(image, bb).final_shape

    result = fitter.fit_from_shape(image, initial_shape, max_iters=40,
                                   camera_update=True,
                                   focal_length_update=False,
                                   reconstruction_weight=1,
                                   shape_prior_weight=.4e8,
                                   texture_prior_weight=1.,
                                   landmarks_prior_weight=1e5,
                                   return_costs=True, init_shape_params_from_lms=False)





    mesh = ColouredTriMesh(result.final_mesh.points, result.final_mesh.trilist)


    def transform(mesh):
        return result._affine_transforms[-1].apply(result.camera_transforms[-1].apply(mesh))


    mesh_in_img = transform(lambertian_shading(mesh))
    expr_dir = image.path.parent
    p = image.path.stem
    raster = rasterize_mesh(mesh_in_img, image.shape)

    uv_shape = (600, 1000)
    template = shape_model.mean()
    unwrapped_template = optimal_cylindrical_unwrap(template).apply(template)

    minimum = unwrapped_template.bounds(boundary=0)[0]
    unwrapped_template = Translation(-minimum).apply(unwrapped_template)
    unwrapped_template.points = unwrapped_template.points[:, [1, 0]]
    unwrapped_template.points[:, 0] = unwrapped_template.points[:, 0].max() - unwrapped_template.points[:, 0]
    unwrapped_template.points *= np.array([.40, .31])
    unwrapped_template.points *= np.array([uv_shape])

    bcoords_img, tri_index_img = rasterize_barycentric_coordinate_images(unwrapped_template, uv_shape)
    TI = tri_index_img.as_vector()
    BC = bcoords_img.as_vector(keep_channels=True).T

    def masked_texture(mesh_in_image, background):

        sample_points_3d = mesh_in_image.project_barycentric_coordinates(BC, TI)

        texture = bcoords_img.from_vector(background.sample(sample_points_3d.points[:, :2]))

        return texture


    uv = masked_texture(mesh_in_img, image)

    t = TexturedTriMesh(result.final_mesh.points, image_coords_to_tcoords(uv.shape).apply(unwrapped_template).points , uv, mesh_in_img.trilist)

    m3io.export_textured_mesh(t, str(expr_dir / Path(p).with_suffix('.mesh.obj')), overwrite=True)
    mio.export_image(raster, str(expr_dir / Path(p).with_suffix('.render.jpg')), overwrite=True)

# In[ ]:




# In[ ]:




# In[ ]:

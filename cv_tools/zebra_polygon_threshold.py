import cv2
import numpy as np


def create_mask(original_img):
    original_img = original_img.transpose(1, 0, 2)
    channel1_min = 1.275
    channel1_max = 98.846
    channel2_min = -36.499
    channel2_max = 30.554
    channel3_min = -28.730
    channel3_max = 57.462
    shift_vec = np.array([-13.572618 ,2.808673 ,40.773466])
    t_mat = np.array([[0.010652, -0.003833, -0.000000, -0.231981],
                 [0.003861, 0.005920, 0.005642, -0.831512],
                 [0.003424, 0.005249, -0.006363, 7.533764],
                 [0.000000, 0.000000, 0.000000, 1.000000]])
    h_points_data = np.array([[-0.391988, -0.481373],
                    [-0.337432, -0.439466],
                    [-0.321465, -0.483578],
                    [-0.288199, -0.505634],
                    [-0.262918, -0.470345],
                    [-0.220338, -0.463728],
                    [-0.002116, -0.540924],
                    [-0.071308, -0.635765],
                    [-0.089937, -0.759278],
                    [-0.089937, -0.774718],
                    [-0.139170, -0.765895],
                    [-0.208362, -0.754867],
                    [-0.256264, -0.752662],
                    [-0.298844, -0.732811],
                    [-0.341424, -0.715166],
                    [-0.377351, -0.637970],
                    [-0.435898, -0.503429]], dtype=np.float32)
    h_points = h_points_data.reshape((-1, 1, 2))
                

    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] = lab[:, :, 0] * (100 / 255)
    lab[:, :, 1] -= 128
    lab[:, :, 2] -= 128
    #print("lab:",lab)
    slider_bw = ((lab[:, :, 0] >= channel1_min) & (lab[:, :, 0] <= channel1_max) &
                 (lab[:, :, 1] >= channel2_min) & (lab[:, :, 1] <= channel2_max) &
                 (lab[:, :, 2] >= channel3_min) & (lab[:, :, 2] <= channel3_max))
    

    m, n, _ = lab.shape
    lab = np.transpose(lab, (2, 0, 1))
    lab = np.reshape(lab, (3, m * n)).T
    poly_bw = np.zeros((m, n), dtype=bool)
    #lab = np.reshape(lab, (m * n, 3))
    lab[:, [0, 1, 2]] = lab[:, [1, 2, 0]]
    j = rotate_color_space(lab, shift_vec, t_mat)
    poly_bw = apply_polygons(j, poly_bw, h_points)
    #print("ploy_bw ", poly_bw)
    bw = slider_bw & poly_bw
    # Reshape the poly_bw back to the OpenCV format
    bw = np.reshape(bw, (m, n))
    #bw=~bw
    masked_original_img_image = np.where(bw[:, :, None], original_img, 0)

    # Transpose the masked_original_img_image back to the OpenCV format
    masked_original_img_image = masked_original_img_image.transpose(1, 0, 2)

    bw = (bw * 1).astype(np.uint8)
    bw = bw.transpose(1, 0)

    return bw, masked_original_img_image

def rotate_color_space(i, shift_vec, t_mat):

    i = i - shift_vec
    #print("I after shiftVec:", i)
    i = np.hstack((i, np.ones((i.shape[0], 1))))
    j = np.dot(t_mat, i.T).T

    return j

def apply_polygons(j, poly_bw, h_points):
    in_poly = np.array([cv2.pointPolygonTest(h_points, (pt[0], pt[1]), False) >= 0 for pt in j])
    in_poly = np.reshape(in_poly, poly_bw.shape)
    poly_bw = poly_bw | in_poly
    return poly_bw


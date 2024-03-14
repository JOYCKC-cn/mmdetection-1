import cv2
import numpy as np


def create_mask(original_img):
    original_img = original_img.transpose(1, 0, 2)
    channel1_min = 0
    channel1_max = 75.594
    channel2_min = -14.990
    channel2_max = 31.903
    channel3_min = -11.961
    channel3_max = 63.057
    shift_vec = np.array([-20.585420, -6.378538, 44.863705])
    t_mat = np.array([
        [0.009298, -0.001565, -0.000000, -0.379412],
        [0.001775, 0.006027, 0.005735, -0.768926,],
        [0.001065, 0.003617, -0.009556, 6.088322,],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])
    h_points_data = np.array([
        [-0.249257, -0.277159],
        [-0.423986, -0.303041],
        [-0.403780, -0.773240],
        [-0.340782, -0.775397],
        [-0.289670, -0.818535],
        [-0.262332, -0.850888],
        [-0.212409, -0.861672],
        [-0.131582, -0.842260],
        [-0.080470, -0.561866]
    ], dtype=np.float32)
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


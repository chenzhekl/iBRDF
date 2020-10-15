import struct

import numpy as np

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360
N_DIM = (
    BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D // 2
)
RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0


def load_merl(filename):
    with open(filename, "rb") as f:
        dim = struct.unpack("iii", f.read(4 * 3))
        n = dim[0] * dim[1] * dim[2]
        if n != N_DIM:
            raise ValueError("invalid BRDF file")

        data = np.empty(3 * n, dtype=np.float32)
        for i in range(3 * n):
            data[i] = struct.unpack("d", f.read(8))[0]

        # color x theta_h x theta_d x phi_d
        data = data.reshape(
            (
                3,
                BRDF_SAMPLING_RES_THETA_H,
                BRDF_SAMPLING_RES_THETA_D,
                BRDF_SAMPLING_RES_PHI_D // 2,
            )
        )

        return data


def save_merl(data, filename, normalize=False):
    if normalize:
        data = np.reshape(data, [3, -1])
        scale = np.array([1.0, RED_SCALE / GREEN_SCALE, RED_SCALE / BLUE_SCALE])
        scale = np.reshape(scale, [3, 1])
        data *= scale
    data = data.flatten()
    with open(filename, "wb") as f:
        f.write(
            struct.pack(
                "iii",
                BRDF_SAMPLING_RES_THETA_H,
                BRDF_SAMPLING_RES_THETA_D,
                BRDF_SAMPLING_RES_PHI_D // 2,
            )
        )
        for i in range(data.shape[0]):
            f.write(struct.pack("d", data[i]))

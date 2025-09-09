import numpy as np
import tensorflow as tf

# the index is like the gps. it will take the embedded vectors and map it in this index for lookup

# reference dataset
ref_ds = ...
ref_imgs, ref_ids = [] []
for imgs, ids in ref_ds:
    ref_imgs.append(imgs)
    ref_ids.append(ids.numpy())
ref_imgs = tf.concat(ref_imgs, axis=0)
ref_imgs = np.concatenate(ref_ids, axis=0).astype(np.int64)

ref_embs = 




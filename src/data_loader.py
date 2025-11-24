
X_CWT, X_FACE, Y, SUBJECT_IDS, SUBJECT_NAMES = load_dataset_from_tree(
    base_dir, CWT_SHAPE, FACE_SHAPE, EMOTIONS,
    cwt_dirname="CWT", face_dirname="Facial_frames",
    key_mode=KEY_MODE, key_regex=KEY_REGEX, pad_width=PAD_WIDTH, offset=OFFSET
)
print('Subjects:', SUBJECT_NAMES)

# Show a few pairs (no augmentation)
def show_pairs(Xcwt, Xface, Y_onehot, n=3):
    import matplotlib.pyplot as plt
    n = min(n, Xcwt.shape[0])
    for i in range(n):
        idx = np.random.randint(0, Xcwt.shape[0])
        emo = EMOTIONS[int(np.argmax(Y_onehot[idx]))]
        plt.figure(figsize=(6,3)); plt.suptitle(f"Pair preview — {emo}")
        plt.subplot(1,2,1); plt.title("CWT");  plt.imshow(Xcwt[idx].squeeze()); plt.axis('off')
        plt.subplot(1,2,2); plt.title("Face"); plt.imshow(np.clip(Xface[idx],0,1)); plt.axis('off')
        plt.show()

show_pairs(X_CWT, X_FACE, Y, n=3)

face_aug = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(factor=0.1),
    layers.RandomContrast(0.1),
    layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01))
], name='face_aug')

cwt_aug = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05, 0.05),
    layers.RandomTranslation(0.03, 0.03),
    layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01))
], name='cwt_aug')

def make_ds(Xcwt, Xface, Y, batch_size, training=True):
    ds = tf.data.Dataset.from_tensor_slices((Xcwt, Xface, Y))
    if training:
        ds = ds.shuffle(buffer_size=len(Xcwt), seed=SEED, reshuffle_each_iteration=True)
        def aug_map(cwt, face, y):
            cwt  = cwt_aug(cwt, training=True)
            face = face_aug(face, training=True)
            return (cwt, face), y
        ds = ds.map(aug_map, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda cwt, face, y: ((cwt, face), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Preview augmented examples
def show_aug_preview(Xcwt, Xface, Y_onehot, n=2):
    for _ in range(n):
        idx = np.random.randint(0, Xcwt.shape[0])
        ds = make_ds(Xcwt[idx:idx+1], Xface[idx:idx+1], Y_onehot[idx:idx+1], 1, training=True)
        (cwt_aug_img, face_aug_img), _ = next(iter(ds))
        emo = EMOTIONS[int(np.argmax(Y_onehot[idx]))]
        plt.figure(figsize=(6,3)); plt.suptitle(f"Aug preview — {emo}")
        plt.subplot(1,2,1); plt.title("CWT (aug)");  plt.imshow(cwt_aug_img[0].numpy().squeeze()); plt.axis('off')
        plt.subplot(1,2,2); plt.title("Face (aug)");  plt.imshow(face_aug_img[0].numpy());         plt.axis('off')
        plt.show()

print("Showing a few augmented pairs...")
show_aug_preview(X_CWT, X_FACE, Y, n=2)


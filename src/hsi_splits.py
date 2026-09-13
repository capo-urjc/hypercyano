import numpy as np
from sklearn.model_selection import train_test_split

N_FOLDS = 5

# Exact (n_val, n_test) per class per fold.
# pure:  11 images -> 7 train / 2 val / 2 test  (10 images cycle through test, 1 is always in train)
# mixed 6class:  30 images -> 22 train / 4 val / 4 test
# mixed 12class: 10 images ->  6 train / 2 val / 2 test  (10 images cycle through test, all tested)
COUNTS = {
    "6class":  {"pure": (2, 2), "mixed": (4, 4)},
    "12class": {"pure": (2, 2), "mixed": (2, 2)},
}


def make_folds(d, task, seed):
    """
    Image-level 5-fold CV with exact per-class image quotas.

    For each class, images are shuffled once, then split into N_FOLDS non-overlapping
    test blocks of n_test images each. Val images are taken from the block immediately
    after the test block (also non-overlapping). The remainder goes to train.

    With 11 pure images and n_test=2: 5*2=10 images cycle through test across folds;
    the remaining 1 image is always in train. This is by design, not a bug.
    """
    rng = np.random.default_rng(seed)
    folds = [{} for _ in range(N_FOLDS)]
    test_imgs_per_fold  = [[] for _ in range(N_FOLDS)]
    val_imgs_per_fold   = [[] for _ in range(N_FOLDS)]
    train_imgs_per_fold = [[] for _ in range(N_FOLDS)]

    for cls in np.unique(d.y):
        cls_rows = np.where(d.y == cls)[0]
        images   = np.unique(d.image_ids[cls_rows])
        is_pure  = d.is_pure[cls_rows[0]]
        n_val, n_test = COUNTS[task]["pure" if is_pure else "mixed"]

        images = images.copy()
        rng.shuffle(images)

        for k in range(N_FOLDS):
            test_start = k * n_test
            val_start  = test_start + n_test

            test_block  = [images[i % len(images)] for i in range(test_start, test_start + n_test)]
            val_block   = [images[i % len(images)] for i in range(val_start,  val_start  + n_val)]
            used        = set(test_block) | set(val_block)
            train_block = [img for img in images if img not in used]

            assert len(test_block) == n_test, f"cls={cls} fold={k}: expected {n_test} test images, got {len(test_block)}"
            assert len(val_block)  == n_val,  f"cls={cls} fold={k}: expected {n_val} val images, got {len(val_block)}"

            test_imgs_per_fold[k]  += test_block
            val_imgs_per_fold[k]   += val_block
            train_imgs_per_fold[k] += train_block

    for k in range(N_FOLDS):
        folds[k] = {
            "test":  np.where(np.isin(d.image_ids, test_imgs_per_fold[k]))[0],
            "val":   np.where(np.isin(d.image_ids, val_imgs_per_fold[k]))[0],
            "train": np.where(np.isin(d.image_ids, train_imgs_per_fold[k]))[0],
        }
    return folds


def split_spectra(y, seed):
    """Fournier et al. spectrum-level 70/15/15 split (leaky baseline, used only by script 01)."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    val, test   = train_test_split(rest, test_size=0.5, random_state=seed, stratify=np.asarray(y)[rest])
    return {"train": train, "val": val, "test": test}

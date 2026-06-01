from src.data.paired_dataset import AANLIB_ROOT, aanlib_split_root, PairedMedicalImageDataset

pairs = ['ct_mri', 'pet_mri', 'spect_mri']
for p in pairs:
    for split in ['train', 'val', 'test']:
        root = aanlib_split_root(str(AANLIB_ROOT), p, split)
        try:
            ds = PairedMedicalImageDataset(root, image_size=256, strict=False, pair=p)
            print(f"{p} {split} {len(ds)}")
        except Exception as e:
            print(f"{p} {split} ERROR {e}")

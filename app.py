import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="dataset/test/train",
    labels_path="dataset/test/train/_annotations.coco.json",
)

session = fo.launch_app(dataset)
session.wait()
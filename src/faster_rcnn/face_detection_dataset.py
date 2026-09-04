class FaceDetectionDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_paths = []

        mode_to_folder = {
            'train': 'train/train',
            'test': 'test/test',
            'valid': 'valid/valid',
        }
        
        folder = mode_to_folder[mode]
        self.img_dir = os.path.join(root, folder)
        csv_path = os.path.join(self.img_dir, '_annotations.csv')
        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['filename'].unique()

        for img_name in self.image_names:
            img_path = os.path.join(self.img_dir, img_name)
            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        img_name = self.image_names[item]

        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        records = self.df[self.df['filename'] == img_name]
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)

        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([item])

        if self.transform:
            image = self.transform(image)
        return image, target
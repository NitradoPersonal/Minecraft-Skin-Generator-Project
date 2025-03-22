import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MinecraftImageDataset(Dataset):
    def __init__(self, folder_path, img_size=64):
        """
        Custom Dataset to load valid images from a directory.
        Args:
            folder_path (str): Path to the folder containing images.
            img_size (int): Target size to resize images (img_size x img_size).
        """
        self.folder_path = folder_path
        self.image_paths = self._filter_valid_images(folder_path)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize images
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        print(f"Total valid images: {len(self.image_paths)}")  # Print the number of valid images

    def _filter_valid_images(self, folder_path):
        """
        Filters out invalid or corrupted images from the directory.
        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            List[str]: List of valid image paths.
        """
        valid_images = []
        for img in os.listdir(folder_path):
            if img.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid extensions
                img_path = os.path.join(folder_path, img)
                try:
                    # Attempt to open and verify the image
                    with Image.open(img_path) as image:
                        image.verify()  # Check if the image is valid
                    valid_images.append(img_path)
                except Exception as e:
                    print(f"Invalid image: {img_path} - {e}")  # Log invalid image details
        return valid_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure 3-channel RGB
        return self.transform(image)  # Apply transformations


def load_minecraft_data(folder_path, img_size=64, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for the Minecraft images with validation.
    Args:
        folder_path (str): Path to the folder containing images.
        img_size (int): Target size to resize images (img_size x img_size).
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
    """
    dataset = MinecraftImageDataset(folder_path, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

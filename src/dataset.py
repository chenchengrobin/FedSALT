
from torchvision import datasets
from torchvision import transforms

from torchvision.transforms.functional import rotate

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

import torch.nn.functional as F

from src.config import *
import os
from functools import lru_cache

# 添加随机种子设置以确保实验可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MNIST_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '.', 'mnist_images'))


class smart_cached_dataset(Dataset):
    def __init__(self, dataset, max_cache_size=13000):
        self.dataset = dataset
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = max_cache_size
        self.current_memory = 0
        self.__dict__.update(dataset.__dict__)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.dataset)}")

        try:
            # with self.lock:
            if idx not in self.cache:
                if len(self.cache) >= self.max_cache_size:
                    while self.cache_order:
                        oldest_idx = self.cache_order.pop(0)
                        if oldest_idx in self.cache:
                            del self.cache[oldest_idx]
                            break

                data = self.dataset[idx]
                self.cache[idx] = data
                self.cache_order.append(idx)

            item = self.cache[idx]
            self.current_memory += item[0].numel() * item[0].element_size()
            if self.current_memory > 2 * 1024 * 1024 * 1024:  # 2GB
                self.clear_cache()

            return item
        except Exception as e:
            self.clear_cache()
            try:
                return self.dataset[idx]
            except Exception as e:
                raise RuntimeError(f"Failed to get item {idx} from dataset: {str(e)}")

    def clear_cache(self):
        self.cache.clear()
        self.cache_order.clear()
        self.current_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class get_global_dataset(Dataset):
    def __init__(self, data_name: str):
        assert data_name in ['MNIST']
        self.data_name = data_name

        if data_name in ['MNIST']:
            self.train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307],
                    std=[0.3081]
                )
            ])

            self.test_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307],
                    std=[0.3081]
                )
            ])

    @lru_cache(maxsize=2)
    def _get_dataset(self, train: bool = True):
        transform = self.train_transform if train else self.test_transform

        if self.data_name == 'MNIST':
            dataset = datasets.MNIST(
                root=MNIST_path,
                train=train,
                download=True,
                transform=transform
            )

        if not isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(dataset.targets)
        dataset.targets = dataset.targets.clone().detach()

        return smart_cached_dataset(dataset)

    @lru_cache(maxsize=1)
    def get_train_dataset(self):
        return self._get_dataset(train=True)

    @lru_cache(maxsize=1)
    def get_test_dataset(self):
        return self._get_dataset(train=False)


class data_distributor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.used_indices = set()

    def distribute_client(self, number, labels=None, balance=True, shuffle=True, exact_indices = None):
        labels = labels if labels is not None else np.arange(10)

        client_indices = []
        for label in labels:
            label_indices = np.where(np.array(self.dataset.targets) == label)[0].tolist()
            label_indices = [idx for idx in label_indices if idx not in self.used_indices]
            if exact_indices is not None:
                label_indices = [idx for idx in label_indices if idx not in exact_indices]

            if not label_indices:
                continue

            num_per_label = number // len(labels) if balance else number
            num_to_take = min(len(label_indices), num_per_label)
            selected = random.choices(label_indices, k=num_to_take)
            client_indices.extend(selected)

        if shuffle:
            random.shuffle(client_indices)
        return Subset(self.dataset, client_indices), client_indices

    def distribute_server(self, number, labels=None, shuffle=True, balance=True):
        labels = labels if labels is not None else np.arange(10)

        server_indices = []
        for label in labels:
            label_indices = np.where(np.array(self.dataset.targets) == label)[0].tolist()

            if not label_indices:
                continue
            
            num_per_label = number // len(labels) if balance else number
            num_to_take = min(len(label_indices), num_per_label)

            selected = random.sample(label_indices, num_to_take)
            server_indices.extend(selected)
            self.used_indices.update(selected)

        if len(server_indices) < number:
            remaining_indices = [i for i in range(len(self.dataset)) if i not in self.used_indices]
            need_more = number - len(server_indices)
            if remaining_indices and need_more > 0:
                extra = random.sample(remaining_indices, min(len(remaining_indices), need_more))
                server_indices.extend(extra)
                self.used_indices.update(extra)

        if shuffle:
            random.shuffle(server_indices)

        data_label = [int(self.dataset.targets[idx]) for idx in server_indices]

        return server_indices, data_label
    
    def add_indices(self, indices):
        self.used_indices.update(indices)

def get_data_all_label(dataset, number, shuffle=True):
    labels = np.arange(CLASS)
    selected_indices = []
    for i, label in enumerate(labels):
        label_indices = np.where(dataset.targets.numpy() == label)[0].tolist()
        indices = random.sample(label_indices, number)
        selected_indices.extend(indices)

    if shuffle:
        random.shuffle(selected_indices)

    dataset = Subset(dataset, selected_indices)

    return dataset

class create_dataset(Dataset):
    def __init__(self, dataset, logit, mode='federated'):
        self.dataset = dataset
        self.logit = logit
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, targets = self.dataset[idx]
        return data, targets, self.logit[idx]


class create_hybrid_dataset(Dataset):
    def __init__(self, dataset, score=None, is_weighted=True):
        self.dataset = dataset
        self.score = score
        self.is_weighted = is_weighted
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = torch.as_tensor(self.dataset[idx][0])
        label = self.dataset[idx][2] if self.score is not None else self.dataset[idx][1]
        if self.is_weighted:
            return self._get_weighted_item(idx, data, label)
        else:
            return self._get_unweighted_item(data, label)
    def _get_weighted_item(self, idx, data, label):
        data = data.to(DEVICE)
        label = torch.as_tensor(label).to(DEVICE)

        if self.score is not None:
            # federated sample
            weight = self.score[idx]
            is_federated_flag = torch.tensor(True).to(DEVICE)
            return data, label, weight, is_federated_flag
        else:
            # privacy sample
            label = F.one_hot(label, CLASS).float().to(DEVICE)
            weight = torch.tensor(1.0).to(DEVICE)
            is_federated_flag = torch.tensor(False).to(DEVICE)
            return data, label, weight, is_federated_flag

        
    def _get_unweighted_item(self, data, label):
        if data.device == 'cpu':
            data = data.to(DEVICE)
        if self.score is not None:
            # federated sample
            label = torch.as_tensor(label)
            weight = torch.tensor(1.0)
            is_federated_flag = torch.tensor(True).to(DEVICE)
            return data, label, weight, is_federated_flag
        else:
            # privacy sample
            label = torch.as_tensor(label)
            label = F.one_hot(label, CLASS).float()
            weight = torch.tensor(1.0)
            is_federated_flag = torch.tensor(False).to(DEVICE)
            return data, label, weight, is_federated_flag
        

def get_dataloader(dataset, shuffle=False, drop_last=True, batch_size=BATCH_SIZE):
    num_workers = 0
    persistent_workers = num_workers > 0 
    prefetch_factor = 2 if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

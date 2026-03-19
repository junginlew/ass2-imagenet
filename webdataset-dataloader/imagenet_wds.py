import os
from typing import Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import webdataset as wds
import albumentations as A
from omegaconf import DictConfig
from aidall_seg.data import BaseDataModule, build_transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TRAIN_TRANSFORMS = [
    A.SmallestMaxSize(max_size=[256, 288, 320, 352, 384, 416, 448, 480]),
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.ToTensorV2(),
]

DEFAULT_VAL_TRANSFORMS = [
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.ToTensorV2(),
]

class ImageNetWDSDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "",
        train_batch_size: int = 16,
        val_batch_size: int = 32,
        train_num_workers: int = 4,
        val_num_workers: int = 4,
        pin_memory: bool = True,
        train_transforms_cfg: Optional[DictConfig] = None,
        val_transforms_cfg: Optional[DictConfig] = None,
        v2: bool = False,
        cutmix_alpha: float = 0.0,
        mixup_alpha: float = 0.0,
        reprob: float = 0.0,
        remode: str = "pixel",
        recount: int = 1,
    ) -> None:
        super().__init__(
            data_dir,
            train_batch_size,
            val_batch_size,
            train_num_workers,
            val_num_workers,
            pin_memory,
            num_classes=1000,
            cutmix_alpha=cutmix_alpha if v2 else 0.0,
            mixup_alpha=mixup_alpha if v2 else 0.0,
            reprob=reprob,
            remode=remode,
            recount=recount,
        )

        self.v2 = v2

        load_dotenv()
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key = os.getenv("S3_ACCESS_KEY_ID")
        self.secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        # Hydra Config가 있으면 build_transforms 사용, 없으면 Default 사용 
        if train_transforms_cfg is not None:
            self.train_transforms = build_transforms(train_transforms_cfg)
        else:
            self.train_transforms = A.Compose(DEFAULT_TRAIN_TRANSFORMS)

        if val_transforms_cfg is not None:
            self.val_transforms = build_transforms(val_transforms_cfg)
        else:
            self.val_transforms = A.Compose(DEFAULT_VAL_TRANSFORMS)

    def _get_s3_presigned_urls(self, prefix: str) -> list:
        s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4')
        )
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        urls = []
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.tar'):
                    url = s3_client.generate_presigned_url(
                        ClientMethod='get_object',
                        Params={'Bucket': self.bucket_name, 'Key': obj['Key']},
                        ExpiresIn=604800 #7일
                    )
                    urls.append(url)
        return urls

    def _apply_albumentations(self, transform_func):
        def wrapper(sample):
            image, label = sample
            augmented = transform_func(image=image)
            return augmented["image"], torch.tensor(label, dtype=torch.long)
        return wrapper

    def _build_wds_pipeline(self, s3_prefix: str, transforms, is_train: bool):
        """
        SeaweedFS(S3) 오브젝트 스토리지로부터 데이터를 실시간으로 스트리밍하여 전처리된 (이미지, 라벨) 튜플을 반환하는 WebDataset 파이프라인을 구축
        """
        urls = self._get_s3_presigned_urls(s3_prefix)
        if not urls:
            raise RuntimeError(f"S3 경로에서 .tar 파일을 찾을 수 없습니다: {s3_prefix}")

        pipeline = [wds.SimpleShardList(urls)]
        
        if is_train:
            pipeline.append(wds.shuffle(100)) # Shard Shuffling

        pipeline.extend([wds.split_by_node, wds.split_by_worker])
        pipeline.append(wds.tariterators.tarfile_to_samples())
        pipeline.append(wds.decode("rgb8"))
        pipeline.append(wds.to_tuple("jpg;jpeg;png", "cls"))
        
        if is_train:
            pipeline.append(wds.shuffle(5000, initial=1000)) # Buffer Shuffling
            
        pipeline.append(wds.map(self._apply_albumentations(transforms)))
        
        return wds.DataPipeline(*pipeline)


    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            train_pipe = self._build_wds_pipeline("imagenet/train", self.train_transforms, True)
            
            # 배치를 묶을 때 BaseDataModule의 MixUp/CutMix collate_fn을 적용
            if self.mixup_cutmix is not None:
                train_pipe.append(
                    wds.batched(self.train_batch_size, collation_fn=self.mixup_cutmix_fn)
                )
            else:
                train_pipe.append(wds.batched(self.train_batch_size))
                
            self.train_dataset = train_pipe

            val_pipe = self._build_wds_pipeline("imagenet/val", self.val_transforms, False)
            val_pipe.append(wds.batched(self.val_batch_size))
            self.val_dataset = val_pipe
        
        if stage == "test" or stage is None:
            test_pipe = self._build_wds_pipeline("imagenet/val", self.val_transforms, False)
            test_pipe.append(wds.batched(self.val_batch_size))
            self.test_dataset = test_pipe
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=None, #batch는 dataset에서 이미 묶여서 나옴
            num_workers=self.train_num_workers, 
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=None, 
            num_workers=self.val_num_workers, 
            pin_memory=self.pin_memory
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=None, 
            num_workers=self.val_num_workers, 
            pin_memory=self.pin_memory
        )

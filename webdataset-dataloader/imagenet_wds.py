import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import webdataset as wds
import albumentations as A

from aidall_seg.data import BaseDataModule  # base.py에 정의된 BaseDataModule

# ImageNet 정규화 상수
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageNetWDSDataModule(BaseDataModule):
    """
    WebDataset 및 SeaweedFS(S3) 스트리밍을 지원하는 ImageNet-1K 데이터모듈.
    DDP 환경에서의 중복 다운로드 방지 및 1Gbps 네트워크 병목 해결을 위한 이중 셔플링 적용.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        load_dotenv()
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key = os.getenv("S3_ACCESS_KEY_ID")
        self.secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(".env 파일에 S3 인증 정보가 올바르게 설정되지 않았습니다.")

        # Albumentations 데이터 증강 파이프라인
        self.train_transforms = A.Compose([
            A.SmallestMaxSize(max_size=[256, 288, 320, 352, 384, 416, 448, 480]),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.ToTensorV2(),
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.ToTensorV2(),
        ])

    def _get_s3_presigned_urls(self, prefix: str) -> list:
        """
        S3 버킷을 순회하며 WebDataset이 접근할 수 있는 7일짜리 보안 임시 URL(Presigned URL) 목록을 반환.
        """
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
                key = obj['Key']
                if key.endswith('.tar'):
                    url = s3_client.generate_presigned_url(
                        ClientMethod='get_object',
                        Params={'Bucket': self.bucket_name, 'Key': key},
                        ExpiresIn=604800  # 7일
                    )
                    urls.append(url)
        return urls

    def _apply_albumentations(self, transform_func):
        """WebDataset 튜플(이미지, 라벨)에 Albumentations 변환을 적용하는 래퍼 함수"""
        def wrapper(sample):
            image, label = sample
            # WebDataset의 decode("rgb8")은 이미지를 numpy 배열로 반환하므로 cv2 기반의 Albumentations와 호환
            augmented = transform_func(image=image)
            return augmented["image"], torch.tensor(label, dtype=torch.long)
        return wrapper

    def _build_wds_pipeline(self, s3_prefix: str, transforms, is_train: bool):
        """WebDataset 스트리밍 파이프라인 구축"""
        urls = self._get_s3_presigned_urls(s3_prefix)
        if not urls:
            raise RuntimeError(f"S3 경로 's3://{self.bucket_name}/{s3_prefix}' 에서 .tar 파일을 찾을 수 없습니다.")

        # URL 목록 리스트
        pipeline = [wds.SimpleShardList(urls)]
        
        # Shard Shuffling (파일 단위 섞기)
        if is_train:
            pipeline.append(wds.shuffle(100))

        # DDP Node/Worker Split
        pipeline.extend([
            wds.split_by_node,
            wds.split_by_worker
        ])

        # tar 파일 열기
        pipeline.append(wds.tariterators.tarfile_to_samples())

        # 디코딩 (jpg -> numpy rgb8) 및 튜플화 (이미지, 라벨)
        pipeline.append(wds.decode("rgb8"))
        pipeline.append(wds.to_tuple("jpg;jpeg;png", "cls"))
        
        # Buffer Shuffling
        if is_train:
            pipeline.append(wds.shuffle(5000, initial=1000))
            
        # Albumentations 전처리 적용
        pipeline.append(wds.map(self._apply_albumentations(transforms)))
        
        return wds.DataPipeline(*pipeline)

    def setup(self, stage: str = None) -> None:
        """PyTorch Lightning 데이터셋 초기화"""
        if stage == "fit" or stage is None:
            # Train 데이터셋 구축 및 배치화
            self.train_dataset = self._build_wds_pipeline(
                s3_prefix="imagenet/train", 
                transforms=self.train_transforms, 
                is_train=True
            ).batched(self.train_batch_size)

            # Val 데이터셋 구축 및 배치화 (Val은 셔플링 생략)
            self.val_dataset = self._build_wds_pipeline(
                s3_prefix="imagenet/val", 
                transforms=self.val_transforms, 
                is_train=False
            ).batched(self.val_batch_size)

    def train_dataloader(self) -> DataLoader:
        """Train 데이터로더 반환"""
        # WebDataset에서 배치를 이미 묶었으므로 batch_size=None으로 설정.
        return DataLoader(
            self.train_dataset, 
            batch_size=None, 
            num_workers=self.train_num_workers, 
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        """Validation 데이터로더 반환"""
        return DataLoader(
            self.val_dataset, 
            batch_size=None, 
            num_workers=self.val_num_workers, 
            pin_memory=self.pin_memory
        )

# ass2
RTX A6000 /project/jeongin/imagenet

Imagenet
- ImageNet-1K 원본 데이터를 webdataset 형태( .tar 파일들의 연속)로 변환
- 변환이 완료된 .tar 파일들을 SeaweedFS S3 버킷에 업로드

Webdataset Dataloader
- S3에서 .tar 파일들을 스트리밍 방식으로 실시간으로 읽어와 전처리 후 텐서로 변환하는 DDP 지원 WebDataset DataLoader 파이프라인 구축

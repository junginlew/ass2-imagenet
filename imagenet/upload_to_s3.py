import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4') # S3 표준 인증 방식
)

def upload_file(local_path, s3_key):
    """단일 파일을 S3에 업로드"""
    try:
        print(f"업로드 시작: {local_path} -> s3://{BUCKET_NAME}/{s3_key}")
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key) # multipart 업로드 자동 지원
        print(f"업로드 완료: {s3_key}")
    except Exception as e:
        print(f"업로드 실패: {s3_key} | 에러: {e}")

def upload_directory_to_s3(local_dir, s3_prefix):
    """폴더 안의 tar 파일들을 병렬로 업로드"""
    if not os.path.exists(local_dir):
        print(f"경로를 찾을 수 없습니다: {local_dir}")
        return

    # 업로드할 파일(.tar) 목록 수집
    upload_tasks = []
    for file_name in os.listdir(local_dir):
        if file_name.endswith('.tar'):
            local_path = os.path.join(local_dir, file_name)         
            s3_key = f"{s3_prefix}/{file_name}"  # S3에 저장될 경로 (예: imagenet/train/imagenet-train-000000.tar)
            upload_tasks.append((local_path, s3_key))

    print(f"[{local_dir}] 총 {len(upload_tasks)}개 파일 업로드 대기 중...")
    with ThreadPoolExecutor(max_workers=4) as executor: # multi threading
        for local_path, s3_key in upload_tasks:
            executor.submit(upload_file, local_path, s3_key)

if __name__ == "__main__":
    print(f"SeaweedFS 버킷 '{BUCKET_NAME}'으로 업로드를 시작합니다.")
    
    upload_directory_to_s3(local_dir="./wds_train", s3_prefix="imagenet/train")
    upload_directory_to_s3(local_dir="./wds_val", s3_prefix="imagenet/val")
    
    print("모든 업로드 작업이 명령되었습니다. 완료될 때까지 기다려주세요.")

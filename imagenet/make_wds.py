import os
import webdataset as wds

def create_webdataset(source_dir, output_dir, prefix, max_size=2e9, max_count=5000): #최대 2GB 또는 5000장
    """
    일반 폴더 구조의 데이터셋을 WebDataset 형식(.tar)으로 변환.
    
    Args:
        source_dir: 원본 이미지가 있는 폴더 (예: './train')
        output_dir: .tar 파일들을 저장할 폴더 (예: './wds_train')
        prefix: 생성될 파일의 접두사 (예: 'imagenet-train')
        max_size: 하나의 tar 파일의 최대 용량 (기본 2GB)
        max_count: 하나의 tar 파일에 들어갈 최대 이미지 수 (기본 5,000장)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 생성될 tar 파일들의 이름 패턴 설정 (예: imagenet-train-000000.tar)
    pattern = os.path.join(output_dir, f"{prefix}-%06d.tar")
    
    # 클래스 폴더 이름 알파벳 순 정렬, 0~999 라벨 부여
    classes = sorted(entry.name for entry in os.scandir(source_dir) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"[{prefix}] 총 {len(classes)}개의 클래스를 찾았습니다. 변환을 시작합니다...")

    with wds.ShardWriter(pattern, maxsize=max_size, maxcount=max_count) as sink:
        for cls_name in classes:
            cls_dir = os.path.join(source_dir, cls_name)
            label = class_to_idx[cls_name]
            
            for img_name in os.listdir(cls_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(cls_dir, img_name)
                key = os.path.splitext(img_name)[0] # 파일 확장자를 제외한 이름 (ex: n01440764_10026)
                
                with open(img_path, "rb") as stream:
                    image_bytes = stream.read() # 이미지를 디코딩하지 않고 binary로 읽음 (속도 최적화)
                
                sample = {
                    "__key__": key,
                    "jpg": image_bytes,
                    "cls": label
                }
                sink.write(sample)
                
    print(f"[{prefix}] WebDataset 변환 완료! ({output_dir} 폴더 확인)")

if __name__ == "__main__":

    create_webdataset(
        source_dir="./train", 
        output_dir="./wds_train", 
        prefix="imagenet-train"
    )
    
    create_webdataset(
        source_dir="./val", 
        output_dir="./wds_val", 
        prefix="imagenet-val"
    )

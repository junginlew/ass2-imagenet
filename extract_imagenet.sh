#!/bin/bash

#  Train 데이터 압축 해제
echo "1. Train 데이터 1차 압축 푸는 중..."
mkdir -p train
# 원본 tar 해제 (클래스별 tar 파일 1000개가 나옴)
tar -xf ILSVRC2012_img_train.tar -C train

echo "2. Train 데이터 클래스별 세부 압축 푸는 중 ..."
cd train
# 각 tar 파일 이름으로 폴더를 만들고, 그 안에 압축을 푼 뒤 tar 파일은 삭제
find . -name "*.tar" | while read NAME ; do
    # NAME 예시: ./n01440764.tar
    DIR_NAME="${NAME%.tar}" # .tar 확장자 제거하여 폴더명 생성
    mkdir -p "${DIR_NAME}"
    tar -xf "${NAME}" -C "${DIR_NAME}"
    rm -f "${NAME}" # 용량 확보를 위해 푼 tar 파일은 즉시 삭제
done
cd ..
echo "Train 데이터 폴더 구조화 완료!"

# 3. Val 데이터 압축 해제
echo "3. Val 데이터 압축 푸는 중..."
mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val
echo "Val 데이터 압축 해제 완료!"

import pandas as pd
import os
import PIL
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score

dogs_directory_train = r'./dogs/train/'
dogs_directory_valid = r'./dogs/valid/'
dogs_directory_test = r'./dogs/test/'


img_path_list = []
labels = []
img_path_list_valid = []
labels_valid = []
img_path_list_test = []
labels_test = []

# os.listdir : 폴더 또는 파일 탐색하는 함수
for folder_name in os.listdir(dogs_directory_train): # 우선 그 폴더를 탐색. dogs_directory는 탐색하는 해당 폴더의 경로
    folder_path = os.path.join(dogs_directory_train, folder_name) # 탐색한 그 폴더의 다음 폴더로 들어갈거니까 join. folder_path : 강아지 이미지가 들어있는 각각의 폴더

    if os.path.isdir(folder_path): # folder_path가 폴더라면. 아니면 false를 반환하고 if문에서 걸리지 않음(파일이라는 뜻)
        for img_name in os.listdir(folder_path): # 해당 path를 탐색하고. path 전체가 불러와진 상태
            img_path = os.path.join(folder_path, img_name)
            what_dog = os.path.basename(folder_path) # folder_path의 제일 마지막 파일 이름만 가지고옴
            
            img_path_list.append(img_path)
            labels.append(what_dog)
            
for folder_name in os.listdir(dogs_directory_valid): # 우선 그 폴더를 탐색. dogs_directory는 탐색하는 해당 폴더의 경로
    folder_path = os.path.join(dogs_directory_valid, folder_name) # 탐색한 그 폴더의 다음 폴더로 들어갈거니까 join. folder_path : 강아지 이미지가 들어있는 각각의 폴더

    if os.path.isdir(folder_path): # folder_path가 폴더라면. 아니면 false를 반환하고 if문에서 걸리지 않음(파일이라는 뜻)
        for img_name in os.listdir(folder_path): # 해당 path를 탐색하고. path 전체가 불러와진 상태
            img_path = os.path.join(folder_path, img_name)
            what_dog = os.path.basename(folder_path) # folder_path의 제일 마지막 파일 이름만 가지고옴
            
            img_path_list_valid.append(img_path)
            labels_valid.append(what_dog)
            

for folder_name in os.listdir(dogs_directory_test): # 우선 그 폴더를 탐색. dogs_directory는 탐색하는 해당 폴더의 경로
    folder_path = os.path.join(dogs_directory_test, folder_name) # 탐색한 그 폴더의 다음 폴더로 들어갈거니까 join. folder_path : 강아지 이미지가 들어있는 각각의 폴더

    if os.path.isdir(folder_path): # folder_path가 폴더라면. 아니면 false를 반환하고 if문에서 걸리지 않음(파일이라는 뜻)
        for img_name in os.listdir(folder_path): # 해당 path를 탐색하고. path 전체가 불러와진 상태
            img_path = os.path.join(folder_path, img_name)
            what_dog = os.path.basename(folder_path) # folder_path의 제일 마지막 파일 이름만 가지고옴
            
            img_path_list_test.append(img_path)
            labels_test.append(what_dog)
            
            
unique_labels = list(set(labels))

# 이미지 경로를 list화 했으면 셔플 후에 train, val, test로 나누기

# 우선 먼저 img list와 label을 한 곳에 묶기
# data = list(zip(img_path_list, labels_encoded))  # zip 사용

# random.shuffle(data) # 섞기

# img_path_list, labels_encoded = zip(*data) # 언패킹 연산자 *

# list 전체 길이를 total_size로 두고
total_size = len(img_path_list)
# 비율대로 size를 나누기
train_size = int(total_size * 0.8)
valid_size = int(total_size * 0.2)
test_size = (total_size - train_size - valid_size) * 0.2

train_list = img_path_list[:train_size]
valid_list = img_path_list[train_size:train_size + valid_size] # 비율로 구해놨어서 더하기 해줘야됨
test_list = img_path_list[train_size:] 

print(f"전체 데이터 개수: {total_size}")
print(f"Train: {len(train_list)}, Test: {len(test_list)}")
print(f"전체 레이블 개수: {len(unique_labels)}개")

class Image_Transform(): 
    def __init__(self,resize,mean,std):
        # 텍스트랑 다르게 이미지는 tensor로 변환하는 과정을 transforms에서 함
        # 따라서 이미지 전처리 한 후에 마지막에 totensor를 해주는게 좋음
        # 아래는 train에 대한 전처리. train과 valid를 구분해줘야함
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                transforms.RandomVerticalFlip(), # 기본값 : p=0.5
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)  # 정규화 추가
                ]),

            # valid는 random resized를 해주지 않는게 좋음. 그래서 resize 또는 centercrop을 이용해주면 됨
            'val': transforms.Compose([
                transforms.Resize(256), # 값 지정해주기. crop size보다 커야함. 안 그러면 crop으로 다 날라가는거니까
                transforms.CenterCrop(resize), # 기본값 : p=0.5
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)  # 정규화 추가
                ]),
            
            'test':transforms.Compose([
                transforms.Resize(256), # 값 지정해주기. crop size보다 커야함. 안 그러면 crop으로 다 날라가는거니까
                transforms.CenterCrop(resize), # 기본값 : p=0.5
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)  # 정규화 추가
                ]),

        }
    def __call__(self,img,data_type): # img 입력, 학습인지 테스트인지 data type 넣어주기
        return self.data_transforms[data_type](img)
    
image_transform = Image_Transform(256,0.5,0.5)

class Dataset(Dataset):
  def __init__(self, file_list, transform=None, data_type=None):
    self.file_list = file_list
    self.transform = transform
    self.data_type = data_type

    self.labels = []
    for img_path in file_list:
        folder_name = os.path.basename(os.path.dirname(img_path))
        self.labels.append(folder_name)

    self.label_encoder = LabelEncoder()
    self.label_encoder.fit(list(set(self.labels)))

    self.label_dict = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}

  
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    img_path = self.file_list[idx]
    img = Image.open(img_path).convert("RGB")  # 이미지를 객체 형태로 변환. ToTensor 적용 가능 
    img_transformed = self.transform(img, self.data_type)

    label = os.path.basename(os.path.dirname(img_path)) # dirname을 넣고 basename을 넣기. img_path는 폴더명이 아닌 파일명이므로 dirname을 해줘야함
    label = self.label_dict[label]

    return img_transformed, torch.tensor(label)

train_dataset = Dataset(train_list, image_transform, 'train')
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

valid_dataset = Dataset(train_list, image_transform, 'val')
valid_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = Dataset(train_list, image_transform, 'test')
test_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=len(unique_labels), pretrained=True):
        super(ResNetClassifier, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        # resnet_param = self.resnet.fc.in_features # 파라미터를 먼저 저장하고
        # self.resnet.fc = nn.Identity() # fc를 무력화함
        # self.linear1 = nn.Linear(resnet_param, num_classes) # 원하는 class수로 정하기. label 수만큼
        # self.new_classifier = nn.Sequential(*list(self.resnet.children())[:-1])
        # self.flatten = nn.Flatten() 
        self.linear1 = nn.Linear(1000, num_classes) 

    def forward(self, x):
        img = self.resnet(x)
        # print(x.size())
        # img = self.flatten(img)
        img = self.linear1(img)
        # print(x.size())
        return img
 
gpu_ids = [0,1,2,3,4]
device = torch.device('cuda:{}'.format(gpu_ids[0]))
model = ResNetClassifier().to(device)
model = nn.DataParallel(model, device_ids=gpu_ids)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

'''
for epoch in range(10):

    total_loss = 0
    correct = 0  
    total_samples = 0  

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) # 학습할 때는 softmax 사용 x. 크로스앤트로피만 사용하기. test할때 softmax 거쳐서 나온 값으로 분류하기
        loss = criterion(outputs, labels)  # 크로스앤트로피 안에 softmax가 있음
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        soft_logit = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)

        # ✅ 정확도 계산
        correct += (prediction == labels).sum().item()  # 예측값과 정답 비교하여 맞춘 개수 저장
        total_samples += labels.size(0)  # 전체 샘플 개수 업데이트

        # print('soft_logit :', soft_logit[0])

    accuracy = correct / total_samples * 100  # 퍼센트(%) 단위 변환
    print(f'예측 label : {prediction[:5].tolist()}')
    print(f'실제 label : {labels[:5].tolist()}')

    print(f'{epoch} 에폭 loss: {total_loss:.4f}, 정확도: {accuracy:.2f}%')

    torch.save(model.state_dict(), f'./dogs/resnet50_dogs2_classification{epoch}.pth')
    print(f'{epoch} 에폭 모델 저장 완료')
'''

learning_rates = [0.001, 0.0001]
batch_sizes = [256, 128]

num_epochs = 50
patience = 5

best_overall_val_loss = float('inf')
best_hyperparams = None
best_model_state = None

# Grid search 결과를 저장할 리스트
grid_search_results = []

for lr, bs in itertools.product(learning_rates, batch_sizes):
    print(f"\n--- Training with learning rate: {lr}, batch size: {bs} ---")
    
    # DataLoader 재설정 (batch_size에 따라)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    
    # 모델 초기화 (모델 재생성)
    model = ResNetClassifier().to(device)
    # DataParallel 사용 시
    model = nn.DataParallel(model, device_ids=gpu_ids)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state_for_this_run = None
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        correct = 0
        total_samples = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss_avg = total_val_loss / len(valid_loader)
        val_accuracy = correct / total_samples * 100
        f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        
        print(f"Epoch {epoch}: Train Loss = {total_train_loss:.4f}, Val Loss = {val_loss_avg:.4f}, Val Acc = {val_accuracy:.2f}%, F1 = {f1:.4f}")
        
        # --- Early Stopping 체크 ---
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # 모델 state를 복사해둡니다.
            best_model_state_for_this_run = model.state_dict()
            print("Best model saved for current hyperparameters.")
        else:
            patience_counter += 1
            print(f"{patience_counter}번 참았다.")
            if patience_counter >= patience:
                print("Early stopping triggered for current hyperparameters!")
                break
                
    grid_search_results.append({
        'learning_rate': lr,
        'batch_size': bs,
        'val_loss': best_val_loss
    })
    
    # 전체 최적 결과와 비교
    if best_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_val_loss
        best_hyperparams = (lr, bs)
        best_model_state = best_model_state_for_this_run

print("\n--- Grid Search Results ---")
for result in grid_search_results:
    print(result)

print(f"\nBest Hyperparameters: Learning Rate = {best_hyperparams[0]}, Batch Size = {best_hyperparams[1]}, with Validation Loss = {best_overall_val_loss:.4f}")

# 최적 모델 저장
torch.save(best_model_state, './dogs/best_resnet_model.pth')
print("Best model state saved to './dogs/best_resnet_model.pth'")

# --- 최종 평가 (Validation) ---
model.load_state_dict(torch.load('./dogs/best_resnet_model.pth'))
model.eval()
total_correct = 0
total_samples = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_accuracy = total_correct / total_samples * 100
final_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\nFinal Validation Accuracy: {final_accuracy:.2f}%")
print(f"Final Validation F1-Score: {final_f1:.4f}")

# --- 테스트 데이터에 대한 이미지 ID 변경 및 CSV 저장 (예시) ---
img_path_list = []
new_img_names = []
labels_list = []  # 필요에 따라 사용할 레이블
counter = 1
for folder_name in sorted(os.listdir(dogs_directory_train)):
    folder_path = os.path.join(dogs_directory_train, folder_name)
    if os.path.isdir(folder_path):
        for img_name in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            
            new_img_name = f"img_{counter:04d}.jpg"
            counter += 1
            
            img_path_list.append(img_path)
            new_img_names.append(new_img_name)
            # 폴더 이름이 레이블이라 가정
            what_dog = os.path.basename(folder_path)
            labels_list.append(what_dog)

import pandas as pd
df_inference = pd.DataFrame({
    'image_id': new_img_names,   # 새 파일명
    'original_path': img_path_list,
    'label': labels_list         # 필요에 따라 사용
})
df_inference.to_csv('test_predictions.csv', index=False)
print("test_predictions.csv 파일에 이미지 ID가 저장되었습니다.")

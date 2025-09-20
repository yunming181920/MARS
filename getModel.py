import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义参数类
class Params:
	def __init__(self):
		self.save_model = True
		self.folder_path = './models'
		self.lr = 0.1
		self.save_on_epochs = [10, 20, 30, 40, 50]

	def to_dict(self):
		return self.__dict__


# 定义模型保存类
class ModelSaver:
	def __init__(self, params):
		self.params = params
		self.best_loss = float('inf')
		if not os.path.exists(self.params.folder_path):
			os.makedirs(self.params.folder_path)

	def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
		torch.save(state, filename)
		if is_best:
			best_filename = filename.replace('.pth.tar', '_best.pth.tar')
			torch.save(state, best_filename)

	def save_model(self, model=None, epoch=0, val_loss=0):
		if self.params.save_model:
			logger.info(f"Saving model to {self.params.folder_path}.")
			model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
			saved_dict = {'state_dict': model.state_dict(),
						  'epoch': epoch,
						  'lr': self.params.lr,
						  'params_dict': self.params.to_dict()}
			self.save_checkpoint(saved_dict, False, model_name)
			if epoch in self.params.save_on_epochs:
				logger.info(f'Saving model on epoch {epoch}')
				self.save_checkpoint(saved_dict, False,
									 filename=f'{self.params.folder_path}/model_epoch_{epoch}.pt.tar')
			if val_loss < self.best_loss:
				self.save_checkpoint(saved_dict, False, f'{model_name}_best')
				self.best_loss = val_loss


# 数据预处理
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 使用预训练的ResNet-18模型
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 10)  # CIFAR-10有10个类别

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


# 训练模型
def train(epoch):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		if batch_idx % 100 == 0:
			logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Accuracy: {100. * correct / total}%')


# 测试模型
def test(epoch, model_saver):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	acc = 100. * correct / total
	logger.info(f'Test Accuracy: {acc}%')
	model_saver.save_model(net, epoch, test_loss / total)
	return acc


# 主函数
def main():
	params = Params()
	model_saver = ModelSaver(params)
	best_acc = 0

	for epoch in range(200):
		train(epoch)
		acc = test(epoch, model_saver)
		if acc > best_acc:
			best_acc = acc
		if best_acc >= 90:
			break


if __name__ == '__main__':
	main()

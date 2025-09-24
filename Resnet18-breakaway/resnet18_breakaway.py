"""resnet18_breakaway.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision
pip install torchsummary

# Usage:
source activate pytorchsenv

# 1.  Full back prop baseline (15 min on a single RTX 4090):
python resnet18_breakaway.py --mode full --epochs 300

# 2.  Greedy breakaway (17 layers x 300 epochs each):
python resnet18_breakaway.py --mode breakaway --sub_epochs 300

# Description (o3 prompt):
implement a SOTA Resnet-18 in pytorch to train on a huggingface image dataset (it must support CIFAR-10). Provide two training options (a and b); 
a) full backpropagation, and;
b) greedy "breakway" learning where;
- each hidden layer l is directly connected to the output layer o via a "breakaway" linear projection, and;
- for each hidden layer l: the l-1 to l and l to o layer weights are independently trained via single hidden layer backprop.

"""

useResnet18 = True	#else resnet9

# -------------------------------------------------------------
#   ResNet-18 (CIFAR-friendly) + two training regimes:
#   1) full back-prop		   (--mode full)
#   2) greedy "breakaway"	   (--mode breakaway)
#
#   Works on any HuggingFace vision dataset whose items are
#   {"img": PIL.Image, "label": int}.  Tested on CIFAR-10.
# -------------------------------------------------------------
import argparse, math, time, pathlib, json
from functools import partial
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torchsummary import summary

# ------------------------------------------------------------------
#  1.  CIFAR-friendly ResNet-18 (BasicBlock identical to He et al.)
# ------------------------------------------------------------------
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
		self.bn1   = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
		self.bn2   = nn.BatchNorm2d(planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes, 1, stride, bias=False),
				nn.BatchNorm2d(planes)
			)
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		return F.relu(out)

# ------------------------------------------------------------------
#  CIFAR-friendly ResNet-18 whose *every conv* has a breakaway head
# ------------------------------------------------------------------

class _GapFeatureHook:
	def __init__(self, model):
		self.model = model

	def __call__(self, _module, _inputs, out):
		# Capture BN activations post-ReLU, mirroring original closure behaviour
		act = F.relu(out)
		self.model._feats.append(self.model.gap(act).flatten(1))


class ResNet18Breakaway(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		# ------- standard ResNet-18 body (unchanged) -------------
		self.in_planes = 64
		self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
		self.bn1   = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(64,  2, 1)
		self.layer2 = self._make_layer(128, 2, 2)
		if(useResnet18):
			self.layer3 = self._make_layer(256, 2, 2)
			self.layer4 = self._make_layer(512, 2, 2)
		self.gap	= nn.AdaptiveAvgPool2d(1)
		if(useResnet18):
			self.fc	 = nn.Linear(512, num_classes)
		else:
			self.fc	 = nn.Linear(128, num_classes)

		# --------------------------------------------------------------
		# collect (conv , bn) pairs in forward order
		self.conv_bn_pairs = [(self.conv1, self.bn1)]
		for blk in self._iter_blocks():
			self.conv_bn_pairs.append((blk.conv1, blk.bn1))
			self.conv_bn_pairs.append((blk.conv2, blk.bn2))

		self.bk_heads = nn.ModuleList(
			[nn.Linear(bn.num_features, num_classes) for _, bn in self.conv_bn_pairs]
		)

		self._register_hooks()

	# -- helper: iterate basic blocks in order --------------------
	def _iter_blocks(self):
		if(useResnet18):
			layers = (self.layer1, self.layer2, self.layer3, self.layer4)
		else:
			layers = (self.layer1, self.layer2)
		for layer in layers:
			for blk in layer:
				yield blk

	# -- standard block builder -----------------------------------
	def _make_layer(self, planes, blocks, stride):
		strides = [stride] + [1]*(blocks-1)
		layers = []
		for s in strides:
			layers.append(BasicBlock(self.in_planes, planes, s))
			self.in_planes = planes
		return nn.Sequential(*layers)

	# -- hook to grab GAP-pooled conv output ----------------------
	def _register_hooks(self):
		self._feats = []
		self._gap_hook = _GapFeatureHook(self)
		self._gap_hook_handles = []
		for _, bn in self.conv_bn_pairs:
			handle = bn.register_forward_hook(self._gap_hook)
			self._gap_hook_handles.append(handle)

	# -- forward --------------------------------------------------
	def _forward_body(self, x):		 # single place with model logic
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		if(useResnet18):
			out = self.layer3(out)
			out = self.layer4(out)
		return out

	def forward(self, x, return_feats=False):
		self._feats = []				# reset per call
		out = self._forward_body(x)
		if return_feats:
			return self._feats		  # list(len = Nº convs) of B×C
		return self.fc(self.gap(out).flatten(1))


# ------------------------------------------------------------------
# 2. HF-dataset -> PyTorch DataLoader helper
# ------------------------------------------------------------------
def build_dataloaders(name, batch, num_workers=4):
	MEAN = [0.4914, 0.4822, 0.4465]
	STD  = [0.2023, 0.1994, 0.2010]
	train_tf = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(), transforms.Normalize(MEAN, STD)
	])
	test_tf  = transforms.Compose([
		transforms.ToTensor(), transforms.Normalize(MEAN, STD)
	])

	def hf_wrap(split, tf):
		ds = load_dataset(name, split=split, trust_remote_code=True)
		class TorchWrap(torch.utils.data.Dataset):
			def __init__(self, hf_ds, tf):
				self.ds, self.t = hf_ds, tf
			def __len__(self): return len(self.ds)
			def __getitem__(self, idx):
				item = self.ds[idx]
				return self.t(item["img"]), item["label"]
		return TorchWrap(ds, tf)

	train_ds = hf_wrap("train", train_tf)
	test_ds  = hf_wrap("test",  test_tf)
	train_ld = DataLoader(train_ds, batch, shuffle=True,
						  num_workers=num_workers, pin_memory=True)
	test_ld  = DataLoader(test_ds,  batch, shuffle=False,
						  num_workers=num_workers, pin_memory=True)
	return train_ld, test_ld

# ------------------------------------------------------------------
#  3.  Training utilities
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, device):
	net.eval()
	correct = total = 0
	for x, y in loader:
		x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
		logits = net(x)
		pred = logits.argmax(1)
		correct += pred.eq(y).sum().item()
		total   += y.size(0)
	return correct / total * 100.0

def train_epoch(net, loader, opt, device, criterion):
	net.train()
	loss_sum = n = 0
	for x, y in loader:
		x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
		opt.zero_grad(set_to_none=True)
		loss = criterion(net(x), y)
		loss.backward()
		opt.step()
		loss_sum += loss.item() * x.size(0)
		n += x.size(0)
	return loss_sum / n

# ------------------------------------------------------------------
#  4-A.  Full back-prop training
# ------------------------------------------------------------------
def train_full(args):
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = ResNet18Breakaway(num_classes=args.num_classes).to(dev)
	print(net)

	train_ld, test_ld = build_dataloaders(args.dataset, args.batch)
	opt = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	scheduler = CosineAnnealingLR(opt, T_max=args.epochs)

	ce = nn.CrossEntropyLoss()
	for ep in range(1, args.epochs+1):
		tr_loss = train_epoch(net, train_ld, opt, dev, ce)
		acc = evaluate(net, test_ld, dev)
		scheduler.step()
		print(f"[full] epoch {ep:3d}/{args.epochs}  "
			  f"train-loss {tr_loss:.4f}  test-acc {acc:.2f}%")
	torch.save(net.state_dict(), "resnet18_full.pth")

# ------------------------------------------------------------------
#  4-B. Greedy breakaway training (conv-level)
# ------------------------------------------------------------------
def train_breakaway(args):
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = ResNet18Breakaway(num_classes=args.num_classes).to(dev)
	print(net)
	train_ld, test_ld = build_dataloaders(args.dataset, args.batch)
	ce = nn.CrossEntropyLoss()

	# freeze everything once
	for p in net.parameters():
		p.requires_grad = False

	# iterate over each conv layer
	for idx, ((conv, bn), head) in enumerate(zip(net.conv_bn_pairs, net.bk_heads)):
		# un-freeze
		for p in (*conv.parameters(), *bn.parameters(), *head.parameters()):
			p.requires_grad = True

		params = list(conv.parameters()) + list(bn.parameters()) + list(head.parameters())
		opt = SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)  # smaller LR
		sch = CosineAnnealingLR(opt, T_max=args.sub_epochs)

		# ------------ sub-epoch loop -----------------------------
		for sub_ep in range(1, args.sub_epochs + 1):
			net.train(); loss_sum = n = 0
			for x, y in train_ld:
				x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
				feats = net(x, return_feats=True)[idx]		  # feature after block idx
				opt.zero_grad(set_to_none=True)
				loss = ce(head(feats), y)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(params, 5.0)
				opt.step()
				loss_sum += loss.item() * x.size(0); n += x.size(0)
			sch.step()

			if sub_ep % args.log == 0 or sub_ep == args.sub_epochs:
				acc = evaluate_head(net, idx, head, test_ld, dev)
				print(f"[breakaway] conv {idx:02d}  "
					  f"sub-epoch {sub_ep:3d}/{args.sub_epochs}  "
					  f"loss {loss_sum/n:.4f}  head-acc {acc:.2f}%")

		# re-freeze before moving on
		for p in params:
			p.requires_grad = False

	# optional: copy last head -> main fc
	net.fc.weight.data.copy_(net.bk_heads[-1].weight.data)
	net.fc.bias.data.copy_(net.bk_heads[-1].bias.data)

	final_acc = evaluate(net, test_ld, dev)
	print(f"[breakaway] ** final top-1 acc using main fc = {final_acc:.2f}% **")
	torch.save(net.state_dict(), "resnet18_breakaway.pth")


@torch.no_grad()
def evaluate_head(net, idx, head, loader, dev):
	net.eval(); c=t=0
	for x, y in loader:
		x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
		feat = net(x, return_feats=True)[idx]
		c += head(feat).argmax(1).eq(y).sum().item()
		t += y.size(0)
	return c/t*100.0


# ------------------------------------------------------------------
#  5.  CLI
# ------------------------------------------------------------------
def cli():
	p = argparse.ArgumentParser()
	p.add_argument("--mode", choices=["full", "breakaway"], default="full")
	p.add_argument("--dataset", default="cifar10",
				   help="any HF image dataset with 'img' + 'label'")
	p.add_argument("--batch",  type=int, default=128)
	p.add_argument("--epochs", type=int, default=300)	  # full-bp
	p.add_argument("--sub_epochs", type=int, default=20)   # per-stage in breakaway
	p.add_argument("--lr",	 type=float, default=0.1)
	p.add_argument("--log",	type=int, default=5)
	p.add_argument("--num_classes", type=int, default=10)
	args = p.parse_args()
	if args.mode == "full":
		train_full(args)
	else:
		train_breakaway(args)

if __name__ == "__main__":
	cli()

# food_classification.py
# usage:
# python food_classification.py --data_dir "C:\Users\soham\Desktop\Project\food-11" --train_dir_name training --val_dir_name validation --test_dir_name evaluation --model resnet18 --epochs 12

import os, argparse, random, json
import numpy as np
from tqdm import tqdm

# plotting libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

# reproducibility
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- simple VGG-lite ----------
class VGGLiteRGB(nn.Module):
    def __init__(self, in_ch=3, n_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- training ----------
def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item(); total += xb.size(0)
    return running_loss/total, correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item(); total += xb.size(0)
            all_preds.append(preds.cpu().numpy()); all_labels.append(yb.cpu().numpy())
    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([])
    return running_loss/total, correct/total, np.concatenate(all_preds), np.concatenate(all_labels)


# ---------- plotting ----------
def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    if plt is None: return
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(train_losses,label="train"); plt.plot(val_losses,label="val"); plt.legend(); plt.title("loss")
    plt.subplot(1,2,2); plt.plot(train_accs,label="train"); plt.plot(val_accs,label="val"); plt.legend(); plt.title("accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"training_curves.png"),dpi=200); plt.close()

def plot_confusion(cm, class_names, out_dir):
    if plt is None: return
    plt.figure(figsize=(10,8))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    else:
        plt.imshow(cm); plt.colorbar()
    plt.xlabel("pred"); plt.ylabel("true"); plt.title("confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"confusion_matrix.png"),dpi=200); plt.close()


# ---------- utils ----------
def find_folder(base_dir, candidates):
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.isdir(p): return c
    return None


def main(args):
    from torchvision.models import resnet18, ResNet18_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size*1.1)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # datasets
    base = args.data_dir
    train_name = args.train_dir_name or find_folder(base, ["train","training"])
    val_name   = args.val_dir_name   or find_folder(base, ["val","validation"])
    test_name  = args.test_dir_name  or find_folder(base, ["test","evaluation"])
    assert train_name and val_name and test_name, "dataset folders missing"

    train_ds = datasets.ImageFolder(os.path.join(base, train_name), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(base, val_name), transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(base, test_name), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("classes:", train_ds.classes)
    n_classes = len(train_ds.classes)

    # model
    if args.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        if args.finetune_last_block:
            for p in model.layer4.parameters(): p.requires_grad = True
    else:
        model = VGGLiteRGB(in_ch=3, n_classes=n_classes)
    model = model.to(device)

    # optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # save paths
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"best_{args.model}.pth")
    classes_path = os.path.join(args.out_dir, "classes.json")

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    # training loop
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc); val_accs.append(val_acc)
        print(f"[epoch {epoch}/{args.epochs}] train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            with open(classes_path,"w",encoding="utf-8") as f: json.dump(train_ds.classes,f,indent=2)
            print("✅ saved best model & classes.json")

    # test
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path,map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"final test acc: {test_acc:.4f}")

    # reports
    plot_curves(train_losses,val_losses,train_accs,val_accs,args.out_dir)
    if preds.size > 0:
        cm = confusion_matrix(labels,preds)
        plot_confusion(cm, train_ds.classes,args.out_dir)
        print(classification_report(labels,preds,digits=4))

    print("done. outputs saved to", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,required=True)
    parser.add_argument("--train_dir_name",type=str,default=None)
    parser.add_argument("--val_dir_name",type=str,default=None)
    parser.add_argument("--test_dir_name",type=str,default=None)
    parser.add_argument("--out_dir",type=str,default="./food_checkpoints")
    parser.add_argument("--model",type=str,default="resnet18",choices=["resnet18","vgg_lite"])
    parser.add_argument("--finetune_last_block",action="store_true")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--step_size",type=int,default=5)
    parser.add_argument("--gamma",type=float,default=0.5)
    parser.add_argument("--img_size",type=int,default=224)
    parser.add_argument("--num_workers",type=int,default=0)
    args = parser.parse_args()
    main(args)

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os


def get_args():
    parser = argparse.ArgumentParser(description="Train a flower classifier")
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()


def main():
    args = get_args()
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    if args.arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
    else:
        model = models.vgg13(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)

    print(f"Training started on {device}...")
    for epoch in range(args.epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                valid_loss += criterion(outputs, labels).item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch + 1}/{args.epochs}.. "
              f"Train loss: {running_loss / len(train_loader):.3f}.. "
              f"Valid loss: {valid_loss / len(valid_loader):.3f}.. "
              f"Valid accuracy: {accuracy / len(valid_loader):.3f}")
        model.train()

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, args.save_dir)
    print(f"Model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
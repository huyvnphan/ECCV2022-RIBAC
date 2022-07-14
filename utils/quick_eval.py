import torch


def backdoor_eval(my_model, device, dataloader):
    my_model = my_model.to(device)
    my_model.eval()
    correct_clean = 0
    correct_backdoor = 0
    total = 0
    with torch.inference_mode():
        for batch in dataloader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred_clean, pred_backdoor, target = my_model(image, label)
            correct_clean += torch.sum(pred_clean.max(1)[1] == label).item()
            correct_backdoor += torch.sum(pred_backdoor.max(1)[1] == target).item()
            total += image.size(0)
    clean_acc = 100 * correct_clean / total
    backdoor_acc = 100 * correct_backdoor / total
    return clean_acc, backdoor_acc


def clean_eval(my_model, device, dataloader):
    my_model = my_model.to(device)
    my_model.eval()
    correct_clean = 0
    total = 0
    with torch.inference_mode():
        for batch in dataloader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred_clean = my_model(image)
            correct_clean += torch.sum(pred_clean.max(1)[1] == label).item()
            total += image.size(0)
    clean_acc = 100 * correct_clean / total
    return clean_acc

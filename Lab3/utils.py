import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("./data/flickr8k_2/data/flickr8k/test_examples/dog.jpg").convert("RGB"))
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    plt.imshow(test_img1.permute(1,2,0))
    plt.title("Example 1 OUTPUT: " + " ".join(model.CNNtoRNN.caption_image(test_img1.unsqueeze(0), dataset.vocab)))
    plt.show()
    
    test_img2 = transform(
        Image.open("./data/flickr8k_2/data/flickr8k/test_examples/child.jpg").convert("RGB")
    )
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    plt.imshow(test_img2.permute(1,2,0))
    plt.title("Example 2 OUTPUT: "+ " ".join(model.CNNtoRNN.caption_image(test_img2.unsqueeze(0), dataset.vocab)))
    plt.show()
    
    test_img3 = transform(Image.open("./data/flickr8k_2/data/flickr8k/test_examples/bus.png").convert("RGB"))
    print("Example 3 CORRECT: Bus driving by parked cars")
    plt.imshow(test_img3.permute(1,2,0))
    plt.title("Example 3 OUTPUT: " + " ".join(model.CNNtoRNN.caption_image(test_img3.unsqueeze(0), dataset.vocab)))
    plt.show()
    
    test_img4 = transform(
        Image.open("./data/flickr8k_2/data/flickr8k/test_examples/boat.png").convert("RGB")
    )
    print("Example 4 CORRECT: A small boat in the ocean")
    plt.imshow(test_img4.permute(1,2,0))
    plt.title("Example 4 OUTPUT: " + " ".join(model.CNNtoRNN.caption_image(test_img4.unsqueeze(0), dataset.vocab)))
    plt.show()
    
    test_img5 = transform(
        Image.open("./data/flickr8k_2/data/flickr8k/test_examples/horse.png").convert("RGB")
    )
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    plt.imshow(test_img5.permute(1,2,0))
    plt.title("Example 5 OUTPUT: " + " ".join(model.CNNtoRNN.caption_image(test_img5.unsqueeze(0), dataset.vocab)))
    plt.show()
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
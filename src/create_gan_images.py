import argparse
import os.path

import numpy
import skimage.io
import torch

# import torchvision
# import torchvision.datasets
# import torchvision.transforms

# import networks

def parseargs():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--generator_path', type=str, help="Input directory for training images", default="../gan/generator.pt")
    parser.add_argument('-o', '--out_dir', type=str, help="Output directory", default="../demo_out/gan")
    parser.add_argument('-n', '--number', type=int, help="Number images", default=1000)
    parser.set_defaults(quiet=False)
    args = parser.parse_args()
    return args

class Interpolate(torch.nn.Module):
    def __init__(self, size, mode, align_corners=False):
        super().__init__()
        self.interp = torch.nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    def __str__(self):
        return "Interpolate(size=" + str(self.size) + "," + str(self.mode) + "," + "align_corners=" + str(self.align_corners) + ")"
    def __repr__(self):
        return self.__str__()
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x

def _gen_conv_block(in_channels, out_channels, kernel_size, stride, padding):
    block = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
    torch.nn.BatchNorm2d(out_channels),
    torch.nn.ReLU(0.2),
    )
    return block

def _gen_conv_transpose_block(in_channels, out_channels, kernel_size, stride, padding):
    block = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
    torch.nn.BatchNorm2d(out_channels),
    torch.nn.ReLU(0.2),
    )
    return block

def _gen_interp_block(interp_size, features_in, mode='bilinear', align_corners=True):
    block = torch.nn.Sequential(
        Interpolate(size=interp_size, mode=mode, align_corners=align_corners),
        torch.nn.BatchNorm2d(features_in),
        torch.nn.ReLU(0.2),
    )
    return block

class Generator28(torch.nn.Module):
    """
    Generator28-sized network
    """
    def __init__(self, z_dim, img_channels, features_g, num_classes, embed_size):
        super().__init__()
        self.embed = torch.nn.Embedding(num_classes, embed_size)
        self.gen = torch.nn.Sequential(
            # Input shape: batch_size x z_dim x 1 x 1
            _gen_conv_transpose_block(z_dim+embed_size, features_g * 16, kernel_size = 5, stride = 1, padding = 0),
            _gen_interp_block((8,8),features_g*16,mode='bilinear',align_corners=True,),
            _gen_conv_block(features_g * 16, features_g * 8, kernel_size = 3, stride = 1, padding = 1),
            _gen_interp_block((15,15),features_g*8,mode='bilinear',align_corners=True),
            _gen_conv_transpose_block(features_g * 8, features_g * 8, kernel_size = 3, stride = 1, padding = 0),
            _gen_interp_block((30,30),mode='bilinear',align_corners=True,features_in=features_g*8),
            torch.nn.Conv2d(features_g * 8, img_channels, kernel_size = 3, stride = 1, padding = 0, bias=False),
            # 28 x 28
            torch.nn.Tanh()
        )
    def forward(self, x, labels):
        """
        Forward pass
        """
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim = 1)
        return self.gen(x)

def get_generator28(features_g,z_dim = 100, channels_out = 1, num_classes = 4,gen_embedding = 100):
    gen = Generator28(z_dim,channels_out,features_g,num_classes,gen_embedding)
    return gen

def main():
    args = parseargs()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # Model initialization
    FEATURES_G = 16
    NOISE_VECTOR = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_generator28(FEATURES_G)
    model.load_state_dict(torch.load(args.generator_path))
    model.to(device)
    model.eval()
    classes = ("control", "coronal", "metopic", "sagittal")
    # Create images
    for cl_int_label,cl in enumerate(classes):
        noise = torch.randn(args.number, NOISE_VECTOR, 1, 1).to(device)
        labels = torch.tensor([cl_int_label] * args.number).to(device)
        fakes = model(noise, labels).to(device)
        class_dir = os.path.join(args.out_dir,cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for i,fake in enumerate(fakes):
            img = fake.detach().numpy().transpose((1,2,0))[:,:,0]
            new_img = numpy.array(255 * img).astype(numpy.uint8)
            skimage.io.imsave(os.path.join(class_dir, "drawn_" + str(i) + ".png"), new_img)

if __name__ == "__main__":
    main()

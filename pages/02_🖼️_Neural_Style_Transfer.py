"""
Neural Transfer Using PyTorch
=============================

**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_
 
**Edited by**: `Winston Herring <https://github.com/winston6>`_


Streamlit Implementation of Neural Style Transfer
=================================================

**Author**: 'Ryan Zhang <http://github.com/Rhyzhang>'_
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import io



def image_to_byte_array(image: Image) -> bytes:
    """Converts a PIL image to a byte array."""
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Page Configuration
st.set_page_config(
     page_title="Neura Style Transfer",
     page_icon="🚧",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "features"
     }
)

st.title("Neural Style Transfer")

st.markdown("""
    Please suggest any features you would like to see in this app! Go here to submit a feature request: [GitHub](https://github.com/Rhyzhang/IP-DALLE/issues)
""")


col1, col2 = st.columns(2)

# Upload the style and content images
style_image = col1.file_uploader("Upload your style image", type=['Jpeg', 'jpg', 'png'])
content_image = col2.file_uploader("Upload your content image", type=['Jpeg', 'jpg', 'png'])

if style_image and content_image is not None:
    # Log expander
    log_expander = st.expander("📝 Log")

    # Check if device has cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load The Images
    # ----------------------------------------------------------------------

    # desired size of output image
    imsize = 512 if torch.cuda.is_available() else 128 # use small size in no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()]) # transform it into a torch tensor

    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit netowrk's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
    
    style_image = image_loader(style_image)
    content_image = image_loader(content_image)

    # assert style_img.size() == content_img.size(), \
    #     "we need to import style and content images of the same size"

    # Display The Images
    # ~~~~~~~~~~~~~~~~~~
    unloader = transforms.ToPILImage() # reconvert int PIL image
    def imshow(tensor, title=None):
        image = tensor.cpu().clone() # clone the tensor to not do changes on it
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if plt.title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated
    def impil(tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image
    
    # display style image
    style_image_fig = plt.figure()
    plt.axis('off')
    imshow(style_image, title='Style Image')
    col1.pyplot(style_image_fig)

    # display content image
    content_image_fig = plt.figure()
    plt.axis('off')
    imshow(content_image, title='Content Image')
    col2.pyplot(content_image_fig)

    # Loss Functions
    # ----------------------------------------------------------------------
    
    # Content Loss
    # ~~~~~~~~~~~~
    class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()
        
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
    
    # Style Loss
    # ~~~~~~~~~~

    # calculate gram matrix
    def gram_matrix(input):
        a, b, c, d = input.size() # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d) # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t()) # compute the gram product

        # 'normalize' the values of the gram matrix
        # by dividing by the number of elements in each feature maps.
        return G.div(a * b * c * d)
    
    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
    
    # Importing The Model
    # ----------------------------------------------------------------------
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Normalize
    # ~~~~~~~~~~
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # create a module to normalize input image so we can easily put it in a 
    # nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
    
    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                    style_img, content_img, 
                                    content_layers=content_layers_default, 
                                    style_layers=style_layers_default):
        #normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable acess to or list of contetnt/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0 # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)
            
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]

        return model, style_losses, content_losses

    input_image = content_image.clone()
    # input_image_fig = plt.figure()
    # imshow(input_image, title='Input Image')
    # st.pyplot(input_image_fig)


    # Gradient Descent
    # _----------------------------------------------------------------------
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                            content_img, style_img, input_img, num_steps=300,
                            style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        log_expander.write('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        input_img.requires_grad_(True)
        model.requires_grad_(False)
    
        optimizer = get_input_optimizer(input_img)

        log_expander.write('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    log_expander.write("run {}:".format(run))
                    log_expander.write('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    log_expander.write('\n')

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output_image = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_image, style_image, input_image)


    output_image_fig = plt.figure()
    imshow(output_image)
    output_image_fig.savefig('output_image.png', bbox_inches='tight')
    st.pyplot(output_image_fig)

    # btn = st.download_button(
    #          label="Download image",
    #          data=r"../output_image.png",
    #          file_name="output_image.png",
    #          mime="png"
    #         )

else:
    st.warning("No image uploaded")

    st.header("What is Neural Style Transfer?")
    st.markdown("""
        Neural Style Transfer (NST) is an algorithm that given a content image and a style image,
        it will output an image that is stylized from the content image to the style image.
    """)
    st.header("Why do NST for DALLE?")
    st.markdown("""
        Styled images can be usful for expanding the range of the image edits and creation process.
        For example, a stylized image can be used to create a more expressive image:
    """)
    # col1, col2, col3 = st.columns(3)
    # col1.image(r"./images/ZOC/bridge_original.png", caption="From this", use_column_width=True)
    # col2.image(r"./images/ZOC/bridge_cropped.png", caption="To this", use_column_width=True)
    # col3.image(r"./images/ZOC/bridge_final.png", caption="To this", use_column_width=True)






            


